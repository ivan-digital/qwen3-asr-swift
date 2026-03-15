#!/usr/bin/env python3
"""Fix Kokoro CoreML models for CPU (BNNS) compatibility.

The FluidInference Kokoro CoreML models contain one non_zero + scatter_nd
pattern in the iSTFTNet vocoder's CustomSTFT phase correction that uses
data-dependent shapes [?, 3]. This crashes the BNNS CPU backend.

This script patches the model.mil file inside .mlmodelc directories to
replace the non_zero + scatter_nd pattern with an element-wise select op,
which has identical semantics but uses fixed shapes throughout.

The original code (in kokoro/custom_stft.py):
    correction_mask = (imag_out == 0) & (real_out < 0)
    phase[correction_mask] = torch.pi

Traces to (in MIL):
    cast_101 = cast(correction_mask, int32)
    non_zero_0 = non_zero(cast_101)            # ← data-dependent shape [?, 3]
    ... (scatter indices setup) ...
    har_phase = scatter_nd(phase_1, indices, pi) # ← applies pi at non-zero positions

The fix replaces this entire block with:
    pi_val = const(val = 0x1.921fb6p+1)         # π
    har_phase = select(cond = correction_mask, a = pi_val, b = phase_1)

Usage:
    python scripts/fix_kokoro_cpu.py /path/to/kokoro_24_10s.mlmodelc
    python scripts/fix_kokoro_cpu.py /tmp/kokoro-coreml-test/  # patches all models in dir
"""

import re
import sys
import shutil
from pathlib import Path


# Flexible pattern: matches any tensor shape [1, 11, N] for correction_mask → scatter_nd block.
# Captures the shape so the replacement uses the correct dimensions.
OLD_BLOCK_PATTERN = re.compile(
    r'(            tensor<bool, (\[1, 11, \d+\])> correction_mask = logical_and\(.*?\];\n)'
    r'(.*?)'  # Everything in between (non-greedy)
    r'            tensor<fp32, \[1, 11, \d+\]> har_phase = scatter_nd\(data = phase_1.*?\];\n',
    re.DOTALL
)


def _build_replacement(match):
    """Build replacement string preserving the correct tensor shape."""
    correction_line = match.group(1)
    shape = match.group(2)  # e.g. [1, 11, 48001]
    return (
        correction_line
        + '            tensor<fp32, []> har_phase_pi_val = const()[name = tensor<string, []>("har_phase_pi_val"), val = tensor<fp32, []>(0x1.921fb6p+1)];\n'
        + f'            tensor<fp32, {shape}> har_phase = select(cond = correction_mask, a = har_phase_pi_val, b = phase_1)[name = tensor<string, []>("har_phase")];\n'
    )


def patch_model_mil(mil_path: Path) -> bool:
    """Patch a model.mil file to replace non_zero+scatter_nd with select."""
    content = mil_path.read_text()

    if 'non_zero' not in content:
        print(f"  {mil_path.parent.name}: no non_zero op found, skipping")
        return False

    # Verify the pattern exists
    match = OLD_BLOCK_PATTERN.search(content)
    if not match:
        print(f"  {mil_path.parent.name}: WARNING - non_zero found but pattern doesn't match")
        return False

    # Create backup
    backup = mil_path.with_suffix('.mil.bak')
    if not backup.exists():
        shutil.copy2(mil_path, backup)
        print(f"  Backed up to {backup.name}")

    # Apply patch
    new_content = OLD_BLOCK_PATTERN.sub(_build_replacement, content)

    # Verify the fix
    if 'non_zero' in new_content:
        print(f"  {mil_path.parent.name}: ERROR - non_zero still present after patch")
        return False

    if 'har_phase = select(cond = correction_mask' not in new_content:
        print(f"  {mil_path.parent.name}: ERROR - replacement not found in output")
        return False

    mil_path.write_text(new_content)
    print(f"  {mil_path.parent.name}: PATCHED successfully")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_kokoro_cpu.py <path>")
        print("  path: .mlmodelc directory or parent directory containing multiple models")
        sys.exit(1)

    target = Path(sys.argv[1])

    if not target.exists():
        print(f"ERROR: {target} does not exist")
        sys.exit(1)

    # Find all model.mil files
    mil_files = []
    if target.is_file() and target.name == "model.mil":
        mil_files = [target]
    elif target.is_dir():
        if (target / "model.mil").exists():
            mil_files = [target / "model.mil"]
        else:
            # Search subdirectories
            mil_files = list(target.glob("**/model.mil"))

    if not mil_files:
        print(f"No model.mil files found in {target}")
        sys.exit(1)

    print(f"Found {len(mil_files)} model(s) to patch\n")

    patched = 0
    for mil_path in sorted(mil_files):
        if patch_model_mil(mil_path):
            patched += 1

    print(f"\nDone: {patched}/{len(mil_files)} models patched")

    if patched > 0:
        print("\nThe patched models should now work with .cpuOnly compute units.")
        print("CoreML will recompile the MIL on next load.")


if __name__ == "__main__":
    main()
