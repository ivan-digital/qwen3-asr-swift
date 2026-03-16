#!/usr/bin/env python3
"""Convert Qwen3-0.6B to CoreML with stateful KV cache.

Usage:
    pip install torch transformers coremltools numpy
    python scripts/convert_qwen3_chat_coreml.py \
        --hf-model Qwen/Qwen3-0.6B \
        --output models/Qwen3-0.6B-Chat-CoreML \
        --quantize int4
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# NOTE: coremltools is imported lazily (after model load) to avoid a segfault
# caused by symbol conflicts between coremltools and torch during from_pretrained.

# Force unbuffered output so we see prints before a crash
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def make_causal_mask(query_len, key_len, dtype=torch.float32):
    """Create a 4D causal mask [1, 1, query_len, key_len].

    Values: 0.0 = attend, -inf = block (future positions).
    """
    mask = torch.full((query_len, key_len), float('-inf'), dtype=dtype)
    past_len = key_len - query_len
    for q in range(query_len):
        for k in range(key_len):
            if k <= past_len + q:
                mask[q, k] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


class Qwen3ChatWrapper(nn.Module):
    """Wraps Qwen3 for CoreML export with explicit KV cache and causal mask.

    The causal mask is passed as an explicit input (not computed internally)
    so that torch.jit.trace produces a graph that handles variable-length
    sequences correctly via CoreML RangeDim.
    """

    def __init__(self, model, max_seq_len=2048):
        super().__init__()
        self.model = model
        self.config = model.config
        self.max_seq_len = max_seq_len
        self.num_layers = self.config.num_hidden_layers
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads)

    def forward(self, input_ids, position_ids, causal_mask, *kv_states):
        """Forward pass with explicit causal mask and KV cache tensors."""
        from transformers.cache_utils import DynamicCache

        past = DynamicCache()
        for i in range(self.num_layers):
            past.update(kv_states[2 * i], kv_states[2 * i + 1], layer_idx=i)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=causal_mask,
            past_key_values=past,
            use_cache=True,
        )

        logits = outputs.logits
        new_cache = outputs.past_key_values

        new_kv = []
        for i in range(self.num_layers):
            # Support DynamicCache API across transformers versions
            if hasattr(new_cache, 'layers'):
                # transformers >= 4.52: DynamicCache.layers[i].keys/.values
                k = new_cache.layers[i].keys
                v = new_cache.layers[i].values
            elif hasattr(new_cache, 'key_cache'):
                # transformers 4.40-4.51: DynamicCache.key_cache[i]
                k = new_cache.key_cache[i]
                v = new_cache.value_cache[i]
            else:
                # transformers < 4.40: tuple cache
                k, v = new_cache[i]
            new_kv.append(k)
            new_kv.append(v)

        return (logits, *new_kv)


def convert_model(hf_model_id: str, output_dir: str, quantize: str = "int4",
                  max_seq_len: int = 2048):
    print(f"Loading {hf_model_id}...", flush=True)
    hf_config = AutoConfig.from_pretrained(hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    print("Loading weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        dtype=torch.float32,
        attn_implementation="eager",  # avoid SDPA conversion issues
    )
    model.eval()

    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)
    vocab_size = hf_config.vocab_size

    print(f"  Layers: {num_layers}, KV heads: {num_kv_heads}, "
          f"Head dim: {head_dim}, Vocab: {vocab_size}", flush=True)

    wrapper = Qwen3ChatWrapper(model, max_seq_len=max_seq_len)
    wrapper.eval()

    # Trace with seq_len=2, past_len=1 so the causal mask contains both
    # 0.0 (attend) and -inf (block) values. Using seq_len=1 would produce
    # an all-zeros mask, causing the tracer to optimize away the mask
    # addition in attention (scores + 0 → scores), making it dead code.
    seq_len = 2
    past_len = 1
    dummy_input_ids = torch.zeros(1, seq_len, dtype=torch.int32)
    dummy_position_ids = torch.tensor([[past_len, past_len + 1]], dtype=torch.int32)
    dummy_mask = make_causal_mask(seq_len, seq_len + past_len)
    dummy_kv = []
    for _ in range(num_layers):
        dummy_kv.append(torch.zeros(1, num_kv_heads, past_len, head_dim))
        dummy_kv.append(torch.zeros(1, num_kv_heads, past_len, head_dim))

    print("Tracing model...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_input_ids, dummy_position_ids, dummy_mask, *dummy_kv))
    print("Trace complete.", flush=True)

    # Import coremltools AFTER tracing to avoid segfault with torch
    import coremltools as ct

    # Define CoreML inputs (explicit causal_mask bypasses internal mask creation)
    inputs = [
        ct.TensorType(
            name="input_ids",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=max_seq_len))),
            dtype=np.int32,
        ),
        ct.TensorType(
            name="position_ids",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=max_seq_len))),
            dtype=np.int32,
        ),
        ct.TensorType(
            name="causal_mask",
            shape=ct.Shape(shape=(
                1, 1,
                ct.RangeDim(lower_bound=1, upper_bound=max_seq_len),
                ct.RangeDim(lower_bound=1, upper_bound=max_seq_len + 1),
            )),
            dtype=np.float16,
        ),
    ]

    for i in range(num_layers):
        for name in ["key", "value"]:
            inputs.append(ct.TensorType(
                name=f"layer_{i}_{name}_cache",
                shape=ct.Shape(shape=(
                    1, num_kv_heads,
                    ct.RangeDim(lower_bound=1, upper_bound=max_seq_len),
                    head_dim
                )),
                dtype=np.float16,
            ))

    # Define outputs
    outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    for i in range(num_layers):
        for name in ["key", "value"]:
            outputs.append(ct.TensorType(
                name=f"layer_{i}_{name}_cache_out",
                dtype=np.float16,
            ))

    print("Converting to CoreML...", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print("Conversion complete.", flush=True)

    # Quantize
    if quantize == "int4":
        print("Quantizing to INT4...", flush=True)
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=64,
        )
        quant_config = ct.optimize.coreml.OptimizationConfig(
            global_config=op_config
        )
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=quant_config)
    elif quantize == "int8":
        print("Quantizing to INT8...", flush=True)
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
        )
        quant_config = ct.optimize.coreml.OptimizationConfig(
            global_config=op_config
        )
        mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=quant_config)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / "Qwen3Chat.mlpackage"
    print(f"Saving to {model_path}...", flush=True)
    mlmodel.save(str(model_path))

    # Save tokenizer files
    print("Saving tokenizer...", flush=True)
    tokenizer.save_pretrained(str(out_path))

    # Save model config
    chat_config = {
        "model_type": "qwen3",
        "hidden_size": hf_config.hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": hf_config.intermediate_size,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "rope_theta": getattr(hf_config, "rope_theta", 1000000.0),
        "rms_norm_eps": getattr(hf_config, "rms_norm_eps", 1e-6),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "quantization": quantize,
    }
    with open(out_path / "chat_config.json", "w") as f:
        json.dump(chat_config, f, indent=2)

    print(f"\nDone! Output: {out_path}", flush=True)
    print(f"  Model:     {model_path}")
    print(f"  Config:    {out_path / 'chat_config.json'}")
    print(f"  Tokenizer: {out_path / 'tokenizer.json'}")


def upload_to_hf(output_dir: str, repo_id: str):
    """Upload converted model to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    out_path = Path(output_dir)

    print(f"Creating/updating repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True)

    files = list(out_path.rglob("*"))
    files = [f for f in files if f.is_file()]

    print(f"Uploading {len(files)} files...")
    api.upload_folder(
        folder_path=str(out_path),
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen3 to CoreML")
    parser.add_argument("--hf-model", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model ID")
    parser.add_argument("--output", default="models/Qwen3-0.6B-Chat-CoreML",
                        help="Output directory")
    parser.add_argument("--quantize", choices=["none", "int4", "int8"],
                        default="int4", help="Quantization mode")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--upload", type=str, default=None,
                        help="HuggingFace repo ID to upload")
    args = parser.parse_args()

    convert_model(args.hf_model, args.output, args.quantize, args.max_seq_len)

    if args.upload:
        upload_to_hf(args.output, args.upload)
