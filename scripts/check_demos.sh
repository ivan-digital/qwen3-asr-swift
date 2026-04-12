#!/usr/bin/env bash
set -euo pipefail

# Verify demo packages compile. Catches missing imports, dependency errors,
# and Swift API changes that break demo code.
#
# Usage: scripts/check_demos.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ERRORS=0

check_demo() {
    local name="$1"
    local dir="$REPO_ROOT/Examples/$name"

    if [ ! -f "$dir/Package.swift" ]; then
        echo "SKIP: $name (no Package.swift)"
        return
    fi

    echo -n "Building $name... "
    if (cd "$dir" && swift build 2>&1 | tail -3); then
        echo "OK"
    else
        echo "FAILED"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "=== Demo Build Check ==="
check_demo SpeechDemo
check_demo PersonaPlexDemo
check_demo DictateDemo
echo "========================"

if [ $ERRORS -gt 0 ]; then
    echo "FAILED: $ERRORS demo(s) failed to build"
    exit 1
else
    echo "All demos compiled successfully"
fi
