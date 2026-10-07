#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds llama.cpp with the Alveo U50 backend.
#
#   1. clones llama.cpp and checks out the pinned commit
#   2. copies the backend sources (llama.cpp/overlay) into the tree
#   3. applies the patches in llama.cpp/patches
#   4. configures and builds with CMake
#
# Environment variables (all optional):
#   XILINX_XRT     XRT installation with include/ and lib/        (default /opt/xilinx/xrt)
#   OUT            build directory                                 (default <repo>/build/llama.cpp)
#   JOBS           parallel build jobs                             (default: number of CPUs)
#   VITIS_INCLUDE  Vitis HLS include directory (contains ap_int.h); if set, the kernel's C++ model is built
#                  too and GGML_XRT_EMULATE=1 runs everything without a card
#   CMAKE_EXTRA    extra CMake arguments, e.g. "-DGGML_NATIVE=OFF -DGGML_AVX2=ON"
#   LLAMA_CPP_REPO llama.cpp git URL                               (default https://github.com/ggml-org/llama.cpp)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/build/llama.cpp}
XRT=${XILINX_XRT:-/opt/xilinx/xrt}
REPO=${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}
JOBS=${JOBS:-$(nproc)}
# llama.cpp revision the patches were written and tested against.
LLAMA_CPP_COMMIT=c92e806d1c81091c9035edce99c35374da1b465e

[ -f "$XRT/include/xrt/xrt_device.h" ] || { echo "XRT headers not found in $XRT (set XILINX_XRT)" >&2; exit 1; }
for tool in git cmake ninja patch; do
    command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done

SRC=$OUT/src
if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$OUT"
    git clone --quiet "$REPO" "$SRC"
fi
git -C "$SRC" checkout --quiet --detach --force "$LLAMA_CPP_COMMIT"
git -C "$SRC" clean --quiet -fdx

cp -a "$ROOT/llama.cpp/overlay/." "$SRC/"
for patch in "$ROOT"/llama.cpp/patches/*.patch; do
    echo "applying $(basename "$patch")"
    git -C "$SRC" apply "$patch"
done

EMULATE=()
if [ -n "${VITIS_INCLUDE:-}" ]; then
    EMULATE=(-DGGML_XRT_EMULATE_INCLUDE="$VITIS_INCLUDE")
fi

XILINX_XRT=$XRT cmake -S "$SRC" -B "$OUT/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_XRT=ON -DGGML_XRT_GEMVQ_INCLUDE="$ROOT/fpga/src" "${EMULATE[@]}" \
    -DLLAMA_CURL=OFF ${CMAKE_EXTRA:-}
cmake --build "$OUT/build" --target llama-cli llama-bench llama-completion -j"$JOBS"

echo
echo "built: $OUT/build/bin"
