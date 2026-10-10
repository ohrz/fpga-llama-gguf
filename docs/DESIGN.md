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
| 1 | `GEMV` | Q4_0, Q8_0, Q5_K, Q6_K weights times the vector in the activation buffer; optional epilog (see below) |
| 2 | `RMSNORM` | per row, optional weight, optional L2 variant, strided rows, optionally *deferred* (see below) |
| 3, 5 | `ADD`, `MUL` | element-wise |
| 4 | `SILU_MUL` | SwiGLU: `silu(g) * u` |
| 6 | `UNARY` | SiLU, sigmoid, softplus, copy; strided rows |
| 7 | `GDN` | one Gated DeltaNet step (Qwen3.5), optionally with q/k L2 norm, gates and gated output norm |
| 8 | `CONV` | one-token causal convolution (kernel width 4) with optional SiLU |
| 9 | `STATE_LOAD` | starts reading the state of the next `GDN` step onto the chip while the following GEMVs run |
| 10 | `ROPE` | rotary position embedding (NEOX pairing) of the query/key heads; the host supplies the cos/sin table |

The command format is documented in `fpga/src/gemvqc16.h`. A launch with zero commands writes a *feature word*
(GEMV formats and fusions the bitstream supports), which the backend probes at start-up; it uses a fusion only if
the bitstream reports it.

### One pass, several commands

The kernel does not only run commands one by one. A *pass* is one dataflow region with the quantizer, the 16
engines, the result writer and the four DeltaNet units; there is exactly one instance of it in hardware.

- **GEMV groups.** Up to four consecutive GEMVs with the same input run as one pass: the input is quantized once
  and the engines stream the matrices back to back.
- **Epilogs.** A GEMV can carry a SwiGLU pair (`g` and `u` of the FFN), the residual `ADD` or the `CONV` of the
  DeltaNet input; the result writer applies it, and the intermediate vector is never written.
- **Deferred norm.** A deferred `RMSNORM` keeps its row on the chip; the next GEMVs on that row normalize while
  quantizing (same arithmetic as the stored result would have).
- **DeltaNet next to GEMVs.** The `GDN` step computes on an on-chip (URAM) copy of the state. `STATE_LOAD`
  prefetches the next state and the previous one is written back during later GEMV passes; the step itself runs in
  the same pass as the following GEMV (the `z` projection).

Every fused form computes the same values as the plain command sequence.

### GEMV engines

- Row `r` of a matrix lives in engine `r % 16`. All engines work on their rows in lockstep and write one result per
  row to the activation buffer.
- Every weight word (64 bytes) is processed per clock: 4 blocks of 32 weights for Q4_0, 2 blocks for Q8_0. Block
  dot products are exact integer arithmetic. For Q4_0/Q8_0 the product with the two FP16 block scales (weight and
  activation) is formed exactly on the significands and shifted into an 80-bit fixed-point accumulator; the row is
  rounded to FP32 once at the end, so the result does not depend on the summation order.
- The activation vector is quantized on the card exactly like ggml does for the matching weight type (`Q8_0` for
  Q4_0/Q8_0 weights, `Q8_K` for Q5_K/Q6_K weights) and broadcast to all engines.
- K-quants are handled per 256-weight super-block: one metadata word with the scales, then the quant words.

### Weight layout (`gemvqw_layout.h`)

The host splits every GGUF row into groups: one 64-byte word with the FP16 block scales followed by the quant
words (K-quants: a metadata word per super-block; for Q6_K three super-blocks share one metadata word and each adds
a high-bits word to its quant words). Only the arrangement changes; the scale and quant values are the original
bits. The same layout code is used by the backend, which keeps host and kernel in agreement.

### Memory map (`fpga/connectivity.cfg`)

| Port | HBM | Content |
| --- | --- | --- |
| `w0`..`w15` | channel pairs `[2e:2e+1]` | weights of engine `e` (384 MiB arena each) |
| `act` | channel 17 | activation buffer: inputs, results, resident FP32 weights and convolution states (64 MiB) |
| `act_rd` | channel 17 | second read port on the same buffer, for the convolution epilog while the pass writes `act` |
| `cmds` | channel 17 | command list (4 KiB) |
| `st0`..`st3` | channels 19, 21, 23, 25 | DeltaNet states, 512-byte rows striped over the four ports |

## The ggml backend (`llama.cpp/overlay/ggml/src/ggml-xrt`)

- **Supported operations.** The backend claims single-token `MUL_MAT` nodes with supported weight formats, `RMS_NORM`
  and `L2_NORM` (with the weight `MUL` that follows), vector `ADD` and `MUL`, SwiGLU (`GLU`), SiLU / sigmoid /
  softplus (`UNARY`), the causal convolution (`CONCAT` + `SSM_CONV`), `GATED_DELTA_NET`, `ROPE`, `CONT` and the few
  `GET_ROWS` / `CPY` / `SCALE` nodes that move recurrent state. Everything else is left to the CPU backend by the
  ggml scheduler.
- **Command lists.** A contiguous run of claimed nodes becomes one list. Inputs from host memory go to the input
  region of the activation buffer, results stay on the card unless a later CPU split (or a graph output) reads them.
- **Fusions.** While building a list the backend maps node patterns onto the fused commands above (GEMV groups,
  SwiGLU pairs, epilogs, deferred norms, the complete DeltaNet step with its norms and gates, `STATE_LOAD` ahead
  of the next step). `GGML_XRT_FEATURE_MASK` hides fusion bits of the kernel for diagnosis.
- **Plan cache.** The analysis of a split (fusions, skipped nodes, use counts) is stored under a fingerprint of the
  split's structure and reused on the next token; only the commands themselves are rebuilt.
- **Weights** are repacked and uploaded the first time a tensor is used; nothing is cached on disk.
- **Recurrent state.** The DeltaNet states and convolution states are allocated by a special buffer type (see patch
  `0004`) and live only on the card. The CPU cannot touch them by accident: the address range is reserved without
  access rights.
- **Graph order** (patch `0006`) puts all input projections of a layer next to each other so they end up in one list.
- **Verification** (`GGML_XRT_VERIFY_CALLS`) recomputes the checked operations with the reference implementation in
  `gemvq_ref.h` and compares them with the card's results.

## Where the time goes

Measured with `GGML_XRT_TIMING` (72 minus 8 generated tokens, weights already uploaded), reference bitstream
(kernel 100 MHz, HBM 346.6 MHz):

| Model | Per token (`llama-bench`) | Backend | Card busy | Weights per token | Launches per token |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B Q4_0 | 41.4 ms | 37.9 ms | 32.3 ms | 2.51 GB | 14 |
| Qwen3.5-4B Q8_0 | 54.6 ms | 54.6 ms | 49.4 ms | 4.47 GB | 10 |
| Qwen3.5-9B Q4_0 | 63.0 ms | 59.6 ms | 53.0 ms | 4.67 GB | 14 |

"Backend" includes building the lists, copies and launches; the rest of the token time is CPU work outside the
backend (attention core, the Q4_1 matrices, sampling). Per-token time and backend time come from different runs. The 16 engines can stream at most 16 × 64 B × 100 MHz =
102.4 GB/s. Over a whole token the card reaches 78 GB/s (4B Q4_0) to 90 GB/s (4B Q8_0), computed as weights
divided by card time, because the DeltaNet steps, norms and the fill and drain of each pass take time too.

The HBM could deliver about three times more bandwidth than the 16 engine ports use. More engines would use it,
but the design already fills most of the U50 (routed kernel: 56 % of the LUTs and 67 % of the DSPs of the user
budget, routing in the HBM-side SLR is the limit). A 32-engine variant did not route; the sources also compile with
20 engines (`-DGEMVQC16_ENGINES=20`, 31 of 32 HBM ports), which has been checked in C simulation only and was never
built into a bitstream.
