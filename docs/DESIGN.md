# Design notes

## Idea

Token generation is dominated by reading every weight once per token. The U50 has 8 GB of HBM2 with a peak of about
316 GB/s, the host CPU of the test machine about 60 GB/s. The kernel keeps the quantized weights in HBM and streams them through 16
engines, each reading its own HBM port at one 64-byte word per clock. Because the weights are kept in the original
GGUF quantization, the card computes the same model as the CPU, not an approximation of it.

## The kernel (`fpga/src`)

`gemvqc16` is one HLS kernel that executes a **command list** from HBM. A launch runs up to 128 commands in order;
they read and write FP32 vectors in one activation buffer, so a command can consume the result of an earlier one
without involving the host.

| Opcode | Command | Notes |
| --- | --- | --- |
| 1 | `GEMV` | Q4_0, Q8_0, Q5_K, Q6_K weights times the vector in the activation buffer |
| 2 | `RMSNORM` | per row, optional weight, optional L2 variant |
| 3, 5 | `ADD`, `MUL` | element-wise |
| 4 | `SILU_MUL` | SwiGLU: `silu(g) * u` |
| 6 | `UNARY` | SiLU, sigmoid, softplus |
| 7 | `GDN` | one Gated DeltaNet step (Qwen3.5), states in four HBM state ports |
| 8 | `CONV` | one-token causal convolution (kernel width 4) with optional SiLU |

The command format is documented in `fpga/src/gemvqc16.h`. A launch with zero commands writes a *feature word*
(which GEMV formats the bitstream supports), which the backend probes at start-up.

### GEMV engines

- Row `r` of a matrix lives in engine `r % 16`. All engines work on their rows in lockstep and write one result per
  row to the activation buffer.
- Every weight word (64 bytes) is processed per clock: 4 blocks of 32 weights for Q4_0, 2 blocks for Q8_0. Block
  dot products are exact integer arithmetic; the per-block scales are applied in FP32 and the row sum is kept in an
  80-bit fixed-point accumulator, so the result does not depend on the summation order.
- The activation vector is quantized on the card exactly like ggml does for the matching weight type (`Q8_0` for
  Q4_0/Q8_0 weights, `Q8_K` for Q5_K/Q6_K weights) and broadcast to all engines.
- K-quants are handled per 256-weight super-block: one metadata word with the scales, then the quant words.

### Weight layout (`gemvqw_layout.h`)

The host splits every GGUF row into groups: one 64-byte word with the FP16 block scales followed by the quant
words (K-quants: a metadata word, and for Q6_K a high-bits word, per super-block). Only the arrangement changes; the
scale and quant values are the original bits. The same layout code is used by the backend, which keeps host and
kernel in agreement.

### Memory map (`fpga/connectivity.cfg`)

| Port | HBM | Content |
| --- | --- | --- |
| `w0`..`w15` | channel pairs `[2e:2e+1]` | weights of engine `e` (384 MiB arena each) |
| `act` | channel 17 | activation buffer: inputs, results, resident FP32 weights and convolution states (64 MiB) |
| `cmds` | channel 17 | command list (4 KiB) |
| `st0`..`st3` | channels 19, 21, 23, 25 | DeltaNet states, 512-byte rows striped over the four ports |

## The ggml backend (`llama.cpp/overlay/ggml/src/ggml-xrt`)

- **Supported operations.** The backend claims single-token `MUL_MAT` nodes with supported weight formats, `RMS_NORM`
  and `L2_NORM` (with the weight `MUL` that follows), vector `ADD` and `MUL`, SwiGLU (`GLU`), SiLU / sigmoid /
  softplus (`UNARY`), the causal convolution (`CONCAT` + `SSM_CONV`), `GATED_DELTA_NET` and the few `GET_ROWS` / `CPY` /
  `SCALE` nodes that move recurrent state. Everything else is left to the CPU backend by the ggml scheduler.
- **Command lists.** A contiguous run of claimed nodes becomes one list. Inputs from host memory go to the input
  region of the activation buffer, results stay on the card unless a later CPU split (or a graph output) reads them.
- **Weights** are repacked and uploaded the first time a tensor is used; nothing is cached on disk.
- **Recurrent state.** The DeltaNet states and convolution states are allocated by a special buffer type (see patch
  `0004`) and live only on the card. The CPU cannot touch them by accident: the address range is reserved without
  access rights.
- **Graph order** (patch `0006`) puts all input projections of a layer next to each other so they end up in one list.
- **Verification** (`GGML_XRT_VERIFY_CALLS`) recomputes the checked operations with the reference implementation in
  `gemvq_ref.h` and compares them with the card's results.

## Where the time goes

Measured with `GGML_XRT_TIMING` (72 minus 8 generated tokens, weights already uploaded), reference bitstream at
96.1 MHz:

| Model | Per token | Card busy | Host to card and back | CPU work | Launches per token |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B Q4_0 | 44.3 ms | 40.0 ms (2.51 GB of weights) | 2.1 ms | 2.2 ms | 22 |
| Qwen3.5-9B Q4_0 | 70.3 ms | 64.1 ms (4.67 GB of weights) | 2.5 ms | 3.1 ms | 22 |

The 16 engines can stream at most 16 × 64 B × 96.1 MHz = 98 GB/s. Over a whole token the card reaches 63 GB/s (4B)
to 73 GB/s (9B), computed as weights divided by card time, because the DeltaNet steps, norms and the fill and drain
of each GEMV take time too. Attention (8 of the 32 layers) runs on the CPU in these measurements.

The HBM could deliver about three times more bandwidth than the 16 engine ports use. More engines would use it; a
32-engine variant did not route on the U50 (routing congestion) and is not part of this release.
