# fpga-llama-gguf

Faster token generation in [llama.cpp](https://github.com/ggml-org/llama.cpp) on an AMD **Alveo U50**.

A 16-engine FPGA kernel computes the matrix-vector products and the surrounding layer operations of the decode
step directly on the **original GGUF quantization** (Q4_0, Q8_0, Q5_K, Q6_K). Nothing is requantized, so the
model is the same one you run on the CPU. A ggml backend plugs the card into llama.cpp.

> **Pre-release.** Tested on one machine with Qwen3.5-4B and Qwen3.5-9B. Expect rough edges, see
> [Limitations](#limitations).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/benchmark-dark.png">
  <img alt="Token generation speed, CPU only versus Alveo U50: 22.1 versus 15.9 tokens/s for Qwen3.5-4B Q4_0, 16.0 versus 10.8 for Q8_0, 14.2 versus 10.0 for Qwen3.5-9B Q4_0" src="docs/images/benchmark-light.png">
</picture>

## Results

Token generation (`llama-bench`, 128 generated tokens, mean of 3 runs, 16 threads). The CPU is the host CPU of
the card (AMD Ryzen Threadripper 1950X, 16 cores, no AVX-512).

| Model (GGUF) | CPU only | Alveo U50 | Speed-up | For reference: Ryzen 9 7950X |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B Q4_0 | 15.87 | **22.13** | 1.39× | 21.28 |
| Qwen3.5-4B Q8_0 | 10.75 | **15.96** ¹ | 1.48× | 13.46 |
| Qwen3.5-9B Q4_0 | 9.98 | **14.17** | 1.42× | – |

¹ measured with 64 generated tokens. The 7950X column was measured earlier with 8 threads and is not part of the
final measurement series; it only shows where a current desktop CPU stands.

What the numbers mean:

- Decoding is limited by memory bandwidth. The 16 engines stream the weights from HBM at up to 98 GB/s
  (16 engines × 64 bytes × 96 MHz), which is worth about 1.4–1.5× the speed of the Threadripper host.
- All three measurements use the *direct register start* of the kernel (`GGML_XRT_DIRECT_START=1`, see
  [docs/BUILD.md](docs/BUILD.md#direct-start-optional)). Without it every one of the roughly 22 kernel launches per
  token costs about 0.2 ms more, around 10 % of the token time of the 4B Q4_0 model.
- Prompt processing runs on the CPU in both setups and is **not** accelerated, see [Limitations](#limitations).
- The kernel clock of this bitstream is 96.1 MHz: the build asked for 100 MHz and the tool lowered it to meet
  timing.

## How it works

```
llama.cpp  ──►  ggml backend "XRT" (llama.cpp/overlay)  ──►  gemvqc16 kernel on the U50 (fpga/src)
                  repacks each weight tensor once              16 engines, one HBM channel pair each
                  builds command lists per layer               GEMV, RMSNorm, add, SwiGLU, conv, DeltaNet
```

- **One launch per layer.** The backend turns consecutive supported ggml nodes into a command list, writes it to
  the card and starts the kernel once. Activations stay in HBM between commands.
- **GGUF-native weights.** At load time the backend splits every supported weight tensor into scales and quants
  in a word-aligned layout. The bits are the original ones.
- **DeltaNet on the card.** The Qwen3.5 recurrent states live in HBM and the Gated DeltaNet step runs as a kernel
  command.
- **Everything else stays on the CPU:** attention, the prompt, the sampler and tensors in unsupported formats.

More detail: [docs/DESIGN.md](docs/DESIGN.md).

## What you need

| | Tested with |
| --- | --- |
| Card | AMD Alveo U50 (`xcu50-fsvh2104-2-e`, 8 GB HBM2), deployment platform `xilinx_u50_gen3x16_xdma_5_202210_1` |
| Host | Linux (Ubuntu 24.04), x86-64, GCC 13, CMake ≥ 3.14, Ninja, git |
| Runtime | AMD XRT 2.26.0 (upstream commit `11169ba5`) with its `xocl` driver |
| Bitstream build | **Vitis / Vivado 2025.2** (includes Vitis HLS), about 48 GB RAM, about 3 hours |
| llama.cpp | commit **`c92e806d1c81091c9035edce99c35374da1b465e`** (11 July 2026), patched by this repository |

The build script downloads llama.cpp itself and pins the commit above. Other llama.cpp versions need patch updates.

## Quick start

```bash
# 1. llama.cpp with the U50 backend (the patches are applied automatically)
source /opt/xilinx/xrt/setup.sh
scripts/build_llama.sh                      # -> build/llama.cpp/build/bin

# 2. the FPGA bitstream (about 3 hours, needs Vitis/Vivado 2025.2)
scripts/build_xclbin.sh                     # -> build/xclbin/gemvqc16.xclbin

# 3. run (model files: see "Test data")
export GGML_XRT_KERNEL=gemvqc16
export GGML_XRT_XCLBIN=$PWD/build/xclbin/gemvqc16.xclbin
export OMP_WAIT_POLICY=ACTIVE
build/llama.cpp/build/bin/llama-cli -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -no-cnv -n 64 -p "The capital of France is"
build/llama.cpp/build/bin/llama-bench -m models/Qwen3.5-4B-Q4_0.gguf -ngl 99 -p 512 -n 128 -r 3 -t 16

# 4. reproduce the table above (CPU only versus U50 for the three test models)
scripts/benchmark.sh
```

The first run uploads the repacked weights to the card, which takes a while. Check correctness with
`GGML_XRT_VERIFY_CALLS=1000`: the backend then compares the first 1000 offloaded operations with a CPU reference
and prints `XRT-VERIFY: PASS` for each (or fails). Full build instructions, options and troubleshooting:
**[docs/BUILD.md](docs/BUILD.md)**.

You can try the backend without a card: `VITIS_INCLUDE=/opt/Xilinx/2025.2/Vitis/include scripts/build_llama.sh`
builds the kernel's C++ model into the backend and `GGML_XRT_EMULATE=1` runs it on the host (very slow: minutes per
token).

## Test data

These are the exact files the numbers above were measured with. They are not part of this repository; download them
from Hugging Face and check the SHA-256.

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| [`Qwen3.5-4B-Q4_0.gguf`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/blob/main/Qwen3.5-4B-Q4_0.gguf) | 2,583,221,408 | `298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb` |
| [`Qwen3.5-4B-Q8_0.gguf`](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/blob/main/Qwen3.5-4B-Q8_0.gguf) | 4,482,403,488 | `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1` |
| [`Qwen3.5-9B-Q4_0.gguf`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q4_0.gguf) | 5,379,417,312 | `17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24` |

Repositories: [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) and
[unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), quantizations of
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Apache-2.0).

```bash
mkdir -p models
curl -L -o models/Qwen3.5-4B-Q4_0.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_0.gguf
sha256sum models/Qwen3.5-4B-Q4_0.gguf
```

The "Q4_0" files are mixed-format files. The card runs the Q4_0, Q8_0, Q5_K (`ssm_out`) and Q6_K (output head)
weight matrices; the four Q4_1 `ffn_down` tensors stay on the CPU.

## Limitations

- **Only token generation is accelerated.** Prompt processing (batched matrix multiplication) runs on the CPU. For
  Q4_0 files the prompt is about 20–25 % *slower* than CPU-only, because the weights of offloaded layers are not
  repacked for the CPU's batched kernels (4B Q4_0: 51 versus 63 tokens/s at 512 prompt tokens, 9B Q4_0: 27
  versus 35; measured with an earlier bitstream).
- **Small models do not benefit.** The card is left once per layer for attention, so launch and transfer costs
  dominate. Qwen2.5-0.5B ran at 0.3–0.55× the CPU speed.
- **Model size.** The repacked weights have to fit into the HBM arenas (16 × 384 MiB). The 9B Q4_0 model fits; a
  9B Q8_0 model does not.
- **Formats.** Q4_0, Q8_0, Q5_K and Q6_K run on the card. Q4_1 and all other quantizations stay on the CPU.
  Matrix widths must be multiples of 128 (256 for K-quants) and at most 16384.
- **Numerics.** Results match the CPU within tolerance but are not bit-identical (different summation order).
  Greedy text was identical to the CPU for the Q4_0 models; the Q8_0 model diverged after about 14 tokens in one
  test, where two candidate tokens were nearly tied.
- **Hardware coverage.** Developed and tested with one U50 in one host (PCIe link at 5 GT/s x8). Other XRT versions,
  platform versions and cards are untested.
- **Timing.** The reference build met timing at 96.1 MHz, not at the requested 100 MHz. A different tool run may
  land on a slightly different clock.

## Repository layout

| Path | Content |
| --- | --- |
| `fpga/src/` | HLS kernel `gemvqc16` and the headers shared with the host code (weight layout, CPU reference) |
| `fpga/connectivity.cfg` | HBM port assignment of the kernel for the linker |
| `llama.cpp/overlay/` | the new ggml backend (`ggml-xrt`), copied into the llama.cpp tree |
| `llama.cpp/patches/` | patches to llama.cpp, applied in order |
| `scripts/` | `build_llama.sh`, `build_xclbin.sh` (the two build steps) and `benchmark.sh` |
| `docs/` | build guide, design notes, benchmark images |

## License

The code in this repository is licensed under the **Apache License 2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)).
It builds on and modifies llama.cpp (MIT) and needs AMD's XRT, Vitis and Alveo platform under their own licenses;
see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). This project is not affiliated with AMD, Hugging Face,
llama.cpp, unsloth or Alibaba; product names are trademarks of their respective owners. No bitstream, model or
third-party tool is distributed here.
