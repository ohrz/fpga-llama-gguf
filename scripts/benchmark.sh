#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Reproduces the table in the README: llama-bench token generation (128 tokens, 5 runs), CPU only versus the U50.
# Models that are not found are skipped.
#
# Environment variables (all optional):
#   BIN              llama.cpp binaries      (default <repo>/build/llama.cpp/build/bin)
#   MODELS           directory with the GGUFs (default <repo>/models)
#   GGML_XRT_XCLBIN  bitstream               (default <repo>/build/xclbin/gemvqc16.xclbin)
#   THREADS          CPU threads             (default 16)
# Set GGML_XRT_DIRECT_START=1 beforehand to use the direct register start (see docs/BUILD.md).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BIN:-$ROOT/build/llama.cpp/build/bin}
MODELS=${MODELS:-$ROOT/models}
THREADS=${THREADS:-16}
export GGML_XRT_KERNEL=gemvqc16
export GGML_XRT_XCLBIN=${GGML_XRT_XCLBIN:-$ROOT/build/xclbin/gemvqc16.xclbin}
export OMP_WAIT_POLICY=ACTIVE

for name in Qwen3.5-4B-Q4_0 Qwen3.5-4B-Q8_0 Qwen3.5-9B-Q4_0; do
    model=$MODELS/$name.gguf
    [ -f "$model" ] || { echo "skipping $name: $model not found"; continue; }
    echo "== $name, CPU only"
    "$BIN/llama-bench" -m "$model" -dev none -ngl 0 -p 0 -n 128 -r 5 -t "$THREADS"
    echo "== $name, Alveo U50"
    "$BIN/llama-bench" -m "$model" -ngl 99 -p 0 -n 128 -r 5 -t "$THREADS"
done
