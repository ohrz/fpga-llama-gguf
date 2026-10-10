#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds the gemvqc16 bitstream (xclbin) for the Alveo U50 with Vitis/Vivado 2025.2:
#   1. v++ -c   HLS compile of fpga/src/gemvqc16.cpp into gemvqc16.xo           (about 10 minutes)
#   2. v++ -l   link, place and route against the U50 platform                  (about 3 hours)
#
# Environment variables (all optional):
#   VITIS_SETTINGS  Vitis environment script   (default /opt/Xilinx/2025.2/Vitis/settings64.sh)
#   PLATFORM        platform .xpfm file        (default the U50 deployment platform under /opt/xilinx/platforms)
#   OUT             output directory           (default <repo>/build/xclbin)
#   FREQ            kernel clock in MHz        (default 100; v++ lowers a clock automatically if its timing is not met)
#   SYNTH_JOBS      Vivado synthesis jobs      (default 4)
#
# The link needs a lot of memory: plan for 48 GB of RAM. Run it with a memory limit if the machine is shared, e.g.
#   systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 scripts/build_xclbin.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/build/xclbin}
VITIS_SETTINGS=${VITIS_SETTINGS:-/opt/Xilinx/2025.2/Vitis/settings64.sh}
PLATFORM=${PLATFORM:-/opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_5_202210_1/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm}
FREQ=${FREQ:-100}
SYNTH_JOBS=${SYNTH_JOBS:-4}

[ -f "$VITIS_SETTINGS" ] || { echo "Vitis settings script not found: $VITIS_SETTINGS (set VITIS_SETTINGS)" >&2; exit 1; }
[ -f "$PLATFORM" ] || { echo "platform not found: $PLATFORM (set PLATFORM)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$VITIS_SETTINGS"
command -v v++ >/dev/null || { echo "v++ not on PATH after sourcing $VITIS_SETTINGS" >&2; exit 1; }

mkdir -p "$OUT"
cd "$OUT"

v++ -c -t hw --platform "$PLATFORM" -k gemvqc16 -I "$ROOT/fpga/src" \
    --hls.clock "${FREQ}000000:gemvqc16" --hls.pre_tcl "$ROOT/fpga/src/gemvqc16_timing.tcl" \
    -o gemvqc16.xo "$ROOT/fpga/src/gemvqc16.cpp" 2>&1 | tee compile.log

v++ -l -t hw --platform "$PLATFORM" --config "$ROOT/fpga/connectivity.cfg" --kernel_frequency "$FREQ" \
    --vivado.synth.jobs "$SYNTH_JOBS" --vivado.impl.jobs 1 -o gemvqc16.xclbin gemvqc16.xo 2>&1 | tee link.log

echo
echo "built: $OUT/gemvqc16.xclbin"
echo "clocks of the bitstream (the kernel runs on DATA_CLK, the HBM on hbm_aclk; v++ lowers a clock when its timing is missed):"
sed -n '/Scalable Clocks/,/System Clocks/p' gemvqc16.xclbin.info 2>/dev/null || true
