# Build and operation

There are two build items. They are independent:

- the **llama.cpp binaries** with the U50 backend (some minutes)
- the **FPGA bitstream** (3–4 hours).

You need the two items to run on the card.

## 1. Prerequisites

| Component | Version used | Notes |
| --- | --- | --- |
| Alveo U50 card | `xcu50-fsvh2104-2-e`, 8 GB HBM2 | one card. The backend opens device 0 if `GGML_XRT_DEVICE` is not set |
| Platform | `xilinx_u50_gen3x16_xdma_5_202210_1` | deployment platform on the card, development platform in `/opt/xilinx/platforms` (for `v++`) |
| XRT | 2.26.0, upstream commit `11169ba504943e72482836feb6bd13af37350bdf` | headers and `libxrt_coreutil` for the build, `xocl` driver at run time |
| Vitis / Vivado / Vitis HLS | **2025.2** | only for the bitstream and the emulator |
| OS / compiler | Ubuntu 24.04, GCC 13.3, CMake 3.28, Ninja 1.11, git | other Linux toolchains are not tested |
| llama.cpp | `c92e806d1c81091c9035edce99c35374da1b465e` | `scripts/build_llama.sh` downloads this commit |

1. Install the U50 platform and XRT. Use the Alveo U50 documentation of AMD.
2. Make sure that XRT finds the card. `xbutil examine` must show the U50.

   ```bash
   source /opt/xilinx/xrt/setup.sh
   xbutil examine
   ```

The backend uses only the public XRT C++ API (`xrt::device`, `xrt::bo`, `xrt::kernel`, `xrt::ip`). We did not
test other XRT or platform versions. XRT 2.26 marks `load_xclbin` and the `xrt::ip` constructor as deprecated. The
compiler shows warnings for them. These warnings have no effect.

## 2. llama.cpp with the U50 backend

```bash
source /opt/xilinx/xrt/setup.sh            # sets XILINX_XRT
scripts/build_llama.sh
```

The script does these steps:

1. It clones llama.cpp into `build/llama.cpp/src` at the commit above.
2. It copies `llama.cpp/overlay/` (the `ggml-xrt` backend) into the tree.
3. It applies `llama.cpp/patches/*.patch` in sequence.
4. It configures with `-DGGML_XRT=ON` and builds `llama-cli`, `llama-bench` and `llama-completion`.

The binaries are in `build/llama.cpp/build/bin`.

| Variable | Default | Function |
| --- | --- | --- |
| `XILINX_XRT` | `/opt/xilinx/xrt` | XRT installation (`include/`, `lib/`) |
| `OUT` | `build/llama.cpp` | work directory and build directory |
| `JOBS` | all CPUs | parallel build jobs |
| `VITIS_INCLUDE` | not set | Vitis HLS include directory (with `ap_int.h`). Enables the emulator (refer to section 4.5) |
| `CMAKE_EXTRA` | not set | more CMake arguments. Example for an older CPU: `-DGGML_NATIVE=OFF -DGGML_AVX2=ON` |
| `LLAMA_CPP_REPO` | GitHub llama.cpp | git URL or path of a llama.cpp clone |

| Patch | Function |
| --- | --- |
| `0001-ggml-add-xrt-backend` | CMake option `GGML_XRT` and backend registration |
| `0002-ggml-sched-mark-split-outputs` | The scheduler marks the tensors that a later split reads. The backend copies back only these tensors |
| `0003-llama-fused-op-host-fallback` | Keeps the fused Gated DeltaNet operation of llama.cpp when it runs on the CPU next to the card |
| `0004-llama-recurrent-state-on-xrt` | Puts the recurrent states (DeltaNet) on the card |
| `0006-qwen35-graph-order` | Puts all input projections of a layer directly after the norm. They then make one split |

## 3. FPGA bitstream

```bash
scripts/build_xclbin.sh
```

The script runs two steps for the U50 platform. The result is `build/xclbin/gemvqc16.xclbin`.

1. `v++ -c`: HLS compile of `fpga/src/gemvqc16.cpp` with the operator latencies of `fpga/src/gemvqc16_timing.tcl`.
2. `v++ -l`: link, placement and routing.

| Variable | Default | Function |
| --- | --- | --- |
| `VITIS_SETTINGS` | `/opt/Xilinx/2025.2/Vitis/settings64.sh` | Vitis environment script |
| `PLATFORM` | the U50 `.xpfm` in `/opt/xilinx/platforms` | platform file |
| `OUT` | `build/xclbin` | output directory |
| `FREQ` | `100` | requested kernel clock in MHz |
| `SYNTH_JOBS` | `4` | Vivado synthesis jobs |

**CAUTION:** Give the build a memory limit on a shared machine. The implementation needs more than 28 GB of RAM. A
build without a limit can stop other processes.

```bash
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 scripts/build_xclbin.sh
```

Data of the reference build:

| Item | Value |
| --- | --- |
| Duration | 3–4 hours on a desktop CPU with 16 cores (synthesis with 4 jobs, implementation with 1 job) |
| Memory | 48 GB RAM recommended |
| Resources after routing (part of the user budget) | 461 k LUT (64 %), 4,611 DSP (78 %), 849 BRAM (73 %), 185 URAM (29 %) |
| Kernel clock | 94.8 MHz (requested 100 MHz, worst slack −0.542 ns in the DeltaNet stages and the command loop) |
| HBM clock (`hbm_aclk`) | 347.9 MHz (platform value 450 MHz, paths in the HBM interconnect of the platform) |

- The DSP usage is more than the 80 % fit threshold of the platform. Because of this, `v++` shows the warning
  `ACCELERATOR-FIT-01`. The design routes.
- If a clock does not meet the timing, `v++` decreases it. The file `gemvqc16.xclbin.info` shows the final clocks
  (section *Scalable Clocks*: `DATA_CLK` is the kernel clock, `hbm_aclk` the HBM clock). The script also shows them.
  A new run can give different clocks. The README values come from the bitstream with 94.8 MHz and 347.9 MHz.
- The DeltaNet units use 8 floating-point lanes (`GEMVQC_GDN_LANES` in `fpga/src/gemvqc_gdn.h`). 16 lanes give
  bit-identical results and save 2 ms for each token of the 4B model. They need 560 more DSPs (HLS estimate). We did
  not build this variant.

HBM connections (`fpga/connectivity.cfg`):

| Port | HBM channel |
| --- | --- |
| `w0`..`w15` (engines 0–15) | `2e` (even channels 0–30) |
| `w16`..`w19` (engines 16–19) | 1, 3, 5, 7 |
| `act`, `act_rd`, `cmds` | 17 |
| `st0`..`st3` (DeltaNet states) | 19, 21, 23, 25 |

The kernel also compiles with 16 engines (`GEMVQC16_ENGINES` in `fpga/src/gemvqc16.h`). For this variant, connect
engine `e` to the channel pair `[2e:2e+1]` and remove `w16`..`w19`. The backend finds the larger arenas.

## 4. Operation

### 4.1 Run

```bash
source /opt/xilinx/xrt/setup.sh
export GGML_XRT_KERNEL=gemvqc16
export GGML_XRT_XCLBIN=$PWD/build/xclbin/gemvqc16.xclbin
export OMP_WAIT_POLICY=ACTIVE          # the CPU threads stay active while the card works: faster

B=build/llama.cpp/build/bin
$B/llama-cli   -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -no-cnv -n 64 -p "The capital of France is"
$B/llama-bench -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -p 512 -n 128 -r 3 -t 16
# CPU only, same machine:
$B/llama-bench -m models/Qwen3.5-4B-Q4_0.gguf -dev none -ngl 0 -p 512 -n 128 -r 3 -t 16
```

`-ngl 99` selects the card. llama.cpp registers the card as a GPU-class device. The first use uploads the repacked
weights. `llama-bench` does this upload in its warm-up run. The speeds that it shows are steady-state values.

### 4.2 Environment variables

Operation:

| Variable | Function |
| --- | --- |
| `GGML_XRT_KERNEL` | kernel name, `gemvqc16` (default) |
| `GGML_XRT_XCLBIN` | path of the bitstream (default `./gemvqc16.xclbin`) |
| `GGML_XRT_DEVICE` | device index (default `0`) or PCI address, for example `0000:08:00.1` |
| `GGML_XRT_DIRECT_START` | not set: register start if `ert_disable` is 1. `1`: register start is necessary. `0`: XRT start. Refer to section 4.3 |
| `GGML_XRT_GDN` | `0`: the backend does not use the DeltaNet ports (for models without DeltaNet layers) |
| `GGML_XRT_WAIT_TIMEOUT_MS` | maximum time for one kernel launch (default 180000) |
| `GGML_XRT_EMULATE` | `1`: the host runs the C++ model of the kernel (emulator build necessary) |

Measurement and diagnosis:

| Variable | Function |
| --- | --- |
| `GGML_XRT_VERIFY_CALLS` | `n`: compares the first `n` offloaded operations with a CPU reference. Shows `XRT-VERIFY` lines |
| `GGML_XRT_TIMING` | shows the time in graphs, copies to the card, launches, waits and copies back, at exit |
| `GGML_XRT_TRACE` | each list starts with a `TRACE` command. Shows the kernel cycles for each command type at exit. The kernel must have feature bit 14. A list then holds one command less |
| `GGML_XRT_READBACK_ALL` | `1`: copies all results back to the host (default: only the results that a later CPU split reads) |
| `GGML_XRT_FEATURE_MASK` | bit mask for the fusion bits of the kernel. `0` runs each fusion as its plain command sequence |
| `GGML_XRT_PLAN_CACHE` | `0`: analyzes each split again. Does not use the cached analysis |
| `GGML_XRT_PLAN_CHECK` | builds each cached plan again and stops if it is different |
| `GGML_XRT_DEBUG` | shows a detailed log |
| `GGML_XRT_EMULATE_DRY` | the emulator does not run the kernel model. The results are not correct, but the host timing is fast |

### 4.3 Direct start

The XRT command scheduler needs 0.2 ms for each kernel launch. With 10–14 launches for each token, this is 2–3 ms.
The direct start writes the control registers of the compute unit (6 µs) and then polls for completion. On the
test machine, the direct start gives 12–15 % more tokens/s.

The backend uses the direct start if the embedded scheduler of the driver is off (sysfs file `ert_disable` is 1).

1. Load the bitstream one time with the scheduler on (`ert_disable` is 0). Each run of llama.cpp loads it.
2. Get the PCI address from `xbutil examine`.
3. Set `ert_disable` to 1.

   ```bash
   echo 1 | sudo tee /sys/bus/pci/devices/0000:08:00.1/ert_disable
   ```

4. Run llama.cpp as usual. The backend now starts the kernel directly.

**CAUTION:** Set `ert_disable` to 0 before you load a different bitstream. If you do not do this, the driver can
deadlock. The backend does not load a different bitstream while `ert_disable` is 1.

```bash
echo 0 | sudo tee /sys/bus/pci/devices/0000:08:00.1/ert_disable
```

With the direct start, the process has exclusive access to the compute unit.

### 4.4 Check the results

```bash
GGML_XRT_VERIFY_CALLS=1000 build/llama.cpp/build/bin/llama-completion \
    -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -no-cnv -n 16 -p "The capital of France is" 2>&1 | grep XRT-VERIFY | sort | uniq -c | head
```

For each checked operation (GEMV, RMSNorm, add, SwiGLU, convolution, DeltaNet step), the backend shows `PASS` and
the largest absolute difference to the CPU reference. If an operation is outside the tolerance, the process stops
with `FAIL`. The backend checks a fused command as one unit. With `GGML_XRT_VERIFY_CALLS=3000`, 16 generated tokens
and the reference bitstream, all 3,208 checks were `PASS` for each of the three test models.

### 4.5 Emulator (no card)

```bash
VITIS_INCLUDE=/opt/Xilinx/2025.2/Vitis/include scripts/build_llama.sh
GGML_XRT_EMULATE=1 GGML_XRT_KERNEL=gemvqc16 build/llama.cpp/build/bin/llama-completion \
    -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -no-cnv -n 6 -p "The capital of France is"
```

The build compiles `fpga/src/gemvqc16.cpp` as C++ with the Vitis HLS headers. The host runs this model in place of
the card. The emulator needs no bitstream and no card. It is slow: one to two minutes for each token of the 4B
model. Use it to test the backend.

The emulator has 20 engines and 250 MiB arenas, as the card. For 16 engines, add
`CMAKE_EXTRA=-DGGML_XRT_EMULATE_ENGINES=16` to the build command.

## 5. Fault isolation

| Message or symptom | Possible cause and action |
| --- | --- |
| `XRT headers not found in ...` | `XILINX_XRT` is not set. Run `source /opt/xilinx/xrt/setup.sh` |
| `XRT Backend: cannot initialize device 0` | The host does not find the card. Check `xbutil examine`, the `xocl` driver and the permissions |
| `XRT Backend: initialization failed ... xclbin` | The path in `GGML_XRT_XCLBIN` is wrong, or the bitstream is for a different platform |
| `engine memory bank too small for the weight arena` | The HBM connections of the bitstream are different from `fpga/connectivity.cfg` |
| `weight arenas full at tensor ...` | The offloaded weights of the model are larger than the 20 × 250 MiB arenas |
| `XRT Backend: output head on the CPU` | The output head does not fit next to the layers. The output head runs on the CPU (slower, but correct) |
| `ert_disable=1 and another xclbin is loaded` | Set `ert_disable` to 0 before you load a different bitstream |
| `GGML_XRT_DIRECT_START needs ert_disable=1` | Set `ert_disable` to 1 (refer to section 4.3). Load the bitstream before |
| `XRT kernel wait timed out` | The kernel stopped. Increase `GGML_XRT_WAIT_TIMEOUT_MS` or reset the card (`xbutil reset`) |
| Slower than the README values | Set `OMP_WAIT_POLICY=ACTIVE` and `-t` (number of cores). Make sure that the direct start is on. Stop other load on the host |
| `v++ ... ERROR ... platform` | `PLATFORM` is not the `.xpfm` file of the development platform |
| Kernel clock below 100 MHz or `hbm_aclk` below 450 MHz after the link | `v++` decreased a clock to meet the timing (refer to section 3). Check `gemvqc16.xclbin.info` |

We measured the README values on a host with a PCIe link at 5 GT/s x8. The U50 supports Gen3 x16. Gen3 x16 makes
the first weight upload faster. It does not change the token speed.
