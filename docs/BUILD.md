# Building and running

Two things are built independently: the **llama.cpp binaries** with the U50 backend (minutes) and the **FPGA
bitstream** (about 3 hours). Both are needed to run on the card.

## 1. Prerequisites

| Component | Version used | Notes |
| --- | --- | --- |
| Alveo U50 card | `xcu50-fsvh2104-2-e`, 8 GB HBM2 | one card; the backend opens device 0 unless `GGML_XRT_DEVICE` is set |
| Platform | `xilinx_u50_gen3x16_xdma_5_202210_1` | deployment platform flashed to the card and development platform in `/opt/xilinx/platforms` (needed by `v++`) |
| XRT | 2.26.0, upstream commit `11169ba504943e72482836feb6bd13af37350bdf` | headers and `libxrt_coreutil` for the build, `xocl` driver at run time |
| Vitis / Vivado / Vitis HLS | **2025.2** | only for the bitstream build and the optional emulation |
| OS / compiler | Ubuntu 24.04, GCC 13.3, CMake 3.28, Ninja 1.11, git | any recent Linux toolchain should work |
| llama.cpp | `c92e806d1c81091c9035edce99c35374da1b465e` | fetched and pinned by `scripts/build_llama.sh` |

Install the U50 platform and XRT following AMD's Alveo U50 documentation, then check the card:

```bash
source /opt/xilinx/xrt/setup.sh
xbutil examine            # the U50 must be listed
```

Other XRT or platform versions may work (the backend only uses the public XRT C++ API: `xrt::device`, `xrt::bo`,
`xrt::kernel`, `xrt::ip`) but were not tested. XRT 2.26 marks `load_xclbin` and the `xrt::ip` constructor as
deprecated, which produces compiler warnings and is harmless.

## 2. llama.cpp with the U50 backend

```bash
source /opt/xilinx/xrt/setup.sh            # sets XILINX_XRT
scripts/build_llama.sh
```

The script

1. clones llama.cpp into `build/llama.cpp/src` and checks out the pinned commit,
2. copies `llama.cpp/overlay/` (the `ggml-xrt` backend) into the tree,
3. applies `llama.cpp/patches/*.patch` in order,
4. configures with `-DGGML_XRT=ON` and builds `llama-cli`, `llama-bench` and `llama-completion`.

Binaries end up in `build/llama.cpp/build/bin`. Environment variables of the script:

| Variable | Default | Meaning |
| --- | --- | --- |
| `XILINX_XRT` | `/opt/xilinx/xrt` | XRT installation (`include/`, `lib/`) |
| `OUT` | `build/llama.cpp` | working and build directory |
| `JOBS` | all CPUs | parallel build jobs |
| `VITIS_INCLUDE` | unset | Vitis HLS include directory (contains `ap_int.h`); enables the emulator, see below |
| `CMAKE_EXTRA` | unset | extra CMake arguments, e.g. `-DGGML_NATIVE=OFF -DGGML_AVX2=ON` for an older CPU |
| `LLAMA_CPP_REPO` | GitHub llama.cpp | git URL or path of a llama.cpp clone |

The patches (`llama.cpp/patches`) do the following:

| Patch | Purpose |
| --- | --- |
| `0001-ggml-add-xrt-backend` | CMake option `GGML_XRT` and backend registration |
| `0002-ggml-sched-mark-split-outputs` | the scheduler marks tensors that a later split reads, so the backend copies back only those |
| `0003-llama-fused-op-host-fallback` | keeps llama.cpp's fused Gated DeltaNet op enabled when it runs on the CPU next to the card |
| `0004-llama-recurrent-state-on-xrt` | allocates the recurrent states (DeltaNet) on the card |
| `0005-llama-xrt-weights-without-cpu-repack` | keeps Q4_0 weights of offloaded layers in GGUF layout instead of the CPU repack format |
| `0006-qwen35-graph-order` | expands all input projections of a layer right after the norm, so they form one contiguous split |

## 3. FPGA bitstream

```bash
scripts/build_xclbin.sh
```

This runs `v++ -c` (HLS compile of `fpga/src/gemvqc16.cpp`) and `v++ -l` (link, place and route) for the U50
platform and writes `build/xclbin/gemvqc16.xclbin`. Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VITIS_SETTINGS` | `/opt/Xilinx/2025.2/Vitis/settings64.sh` | Vitis environment script |
| `PLATFORM` | the U50 `.xpfm` under `/opt/xilinx/platforms` | platform file |
| `OUT` | `build/xclbin` | output directory |
| `FREQ` | `100` | requested kernel clock in MHz |
| `SYNTH_JOBS` | `4` | Vivado synthesis jobs |

Expect:

- **About 3 hours** on a 16-core desktop CPU (synthesis with 4 jobs, implementation with 1).
- **Memory:** the implementation needs more than 28 GB; plan for 48 GB. On a shared machine, run the script with a
  memory limit, for example `systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 scripts/build_xclbin.sh`.
- **Resources of the reference build** (after routing, share of the user budget): 330 k LUT (46 %), 4,299 DSP (72 %),
  622 BRAM (54 %), 33 URAM.
- **Clock:** the reference build asked for 100 MHz, missed it by 0.4 ns (routing congestion) and the tool lowered
  the kernel clock to **96.1 MHz** (HBM clock 399.6 MHz), where timing is met. The final clocks are listed in
  `gemvqc16.xclbin.info` (section *Scalable Clocks*) and printed at the end of the script. A rerun can end up with a
  slightly different clock; the results in the README were measured at 96.1 MHz.
- The kernel ports are connected as in `fpga/connectivity.cfg`: engine `e` (ports `w0`..`w15`) to HBM channels
  `[2e:2e+1]`, the activation and command buffers to channel 17, the four DeltaNet state ports to channels
  19, 21, 23 and 25.

## 4. Running

```bash
source /opt/xilinx/xrt/setup.sh
export GGML_XRT_KERNEL=gemvqc16
export GGML_XRT_XCLBIN=$PWD/build/xclbin/gemvqc16.xclbin
export OMP_WAIT_POLICY=ACTIVE          # CPU threads keep spinning while the card works: faster

B=build/llama.cpp/build/bin
$B/llama-cli   -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -no-cnv -n 64 -p "The capital of France is"
$B/llama-bench -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -p 512 -n 128 -r 3 -t 16
# CPU-only comparison on the same machine:
$B/llama-bench -m models/Qwen3.5-4B-Q4_0.gguf -dev none -ngl 0 -p 512 -n 128 -r 3 -t 16
```

`-ngl 99` selects the card (it is registered as a GPU-class device). The first use uploads the repacked weights;
`llama-bench` does this in its warm-up run, so the reported speeds are steady-state.

### Environment variables of the backend

| Variable | Meaning |
| --- | --- |
| `GGML_XRT_KERNEL` | kernel name, `gemvqc16` (the default) |
| `GGML_XRT_XCLBIN` | path of the bitstream (default `./gemvqc16.xclbin`) |
| `GGML_XRT_DEVICE` | device index (default `0`) or PCI address such as `0000:08:00.1` |
| `GGML_XRT_DIRECT_START` | `1`: start the kernel through its registers, see below |
| `GGML_XRT_GDN` | `0`: ignore the DeltaNet ports of the kernel (models without DeltaNet layers do not need them) |
| `GGML_XRT_VERIFY_CALLS` | `n`: compare the first `n` offloaded operations with a CPU reference and print `XRT-VERIFY` lines |
| `GGML_XRT_TIMING` | set: print the time spent in graphs, host-to-card copies, launches, waits and copy-back at exit |
| `GGML_XRT_READBACK_ALL` | `1`: copy every result back to the host (default: only what a later CPU split reads) |
| `GGML_XRT_WAIT_TIMEOUT_MS` | upper bound for one kernel launch (default 180000) |
| `GGML_XRT_DEBUG` | set: verbose log |
| `GGML_XRT_EMULATE` | `1`: run the kernel as its C++ model on the host (needs the emulator build) |

### Direct start (optional)

The kernel is normally started through XRT's command scheduler, which costs about 0.2 ms per launch. With about 22
launches per token this is around 10 % of the token time. `GGML_XRT_DIRECT_START=1` writes the control registers of
the compute unit directly (about 6 µs) and polls for completion. It needs the driver's embedded scheduler disabled
through the sysfs file `ert_disable`:

```bash
# 1. load the bitstream once with the scheduler enabled (any run without GGML_XRT_DIRECT_START)
# 2. then, with the PCI address from "xbutil examine":
echo 1 | sudo tee /sys/bus/pci/devices/0000:08:00.1/ert_disable
# 3. run with
export GGML_XRT_DIRECT_START=1
# 4. before loading a different bitstream, switch it back
echo 0 | sudo tee /sys/bus/pci/devices/0000:08:00.1/ert_disable
```

The backend refuses to load a *different* bitstream while `ert_disable` is 1, because the driver would deadlock. In
this mode the process owns the compute unit exclusively.

### Checking correctness

```bash
GGML_XRT_VERIFY_CALLS=1000 build/llama.cpp/build/bin/llama-completion \
    -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -no-cnv -n 16 -p "The capital of France is" 2>&1 | grep XRT-VERIFY | sort | uniq -c | head
```

Every checked operation (GEMV, RMSNorm, add, SwiGLU, convolution, DeltaNet step) prints `PASS` with the largest
absolute difference to the CPU reference, or the process stops with `FAIL`. With the 4B Q4_0 model, 16 generated tokens and
the reference bitstream, all 6,429 checked operations passed.

### Without a card (emulator)

```bash
VITIS_INCLUDE=/opt/Xilinx/2025.2/Vitis/include scripts/build_llama.sh
GGML_XRT_EMULATE=1 GGML_XRT_KERNEL=gemvqc16 build/llama.cpp/build/bin/llama-completion \
    -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -no-cnv -n 6 -p "The capital of France is"
```

This compiles `fpga/src/gemvqc16.cpp` as plain C++ against the Vitis HLS headers and runs it instead of the card. It
needs no bitstream and no hardware but is slow: 6 tokens of the 4B model take about 15 minutes. It is meant for
testing the backend.

## 5. Troubleshooting

| Message / symptom | Likely cause |
| --- | --- |
| `XRT headers not found in ...` | `XILINX_XRT` not set; `source /opt/xilinx/xrt/setup.sh` |
| `XRT Backend: cannot initialize device 0` | card not visible: check `xbutil examine`, the `xocl` driver and permissions |
| `XRT Backend: initialization failed ... xclbin` | wrong path in `GGML_XRT_XCLBIN`, or a bitstream built for another platform |
| `engine memory bank too small for the weight arena` | the bitstream's HBM connectivity differs from `fpga/connectivity.cfg` |
| `weight arenas full at tensor ...` | the model's offloadable weights do not fit into the 16 × 384 MiB arenas |
| `ert_disable=1 and another xclbin is loaded` | switch `ert_disable` back to 0 before loading a different bitstream |
| `GGML_XRT_DIRECT_START needs ert_disable=1` | set `ert_disable` as described above (and load the bitstream once before) |
| `XRT kernel wait timed out` | kernel hang; raise `GGML_XRT_WAIT_TIMEOUT_MS` or reset the card (`xbutil reset`) |
| slower than the table in the README | `OMP_WAIT_POLICY=ACTIVE` and `-t` (number of cores) set? direct start enabled? other load on the host? |
| `v++ ... ERROR ... platform` | `PLATFORM` does not point to the development platform's `.xpfm` |
| link ends with a clock below 100 MHz | expected for this design, see "Clock" above; check `gemvqc16.xclbin.info` |

The numbers in the README were measured on a host whose PCIe link ran at 5 GT/s x8. The U50 supports Gen3 x16, which
shortens the one-time weight upload but should not change the token rate in steady state.
