# Design description

## 1. Principle

Token generation reads each weight one time for each token. The U50 has 8 GB of HBM2 with a peak bandwidth of
316 GB/s. The host CPU of the test machine has 60 GB/s.

The kernel keeps the quantized weights in HBM. 20 engines read them. Each engine reads its own HBM port at one
64-byte word for each clock. The weights stay in the original GGUF quantization. Because of this, the card
calculates the same model as the CPU.

## 2. Kernel (`fpga/src`)

### 2.1 Command list

`gemvqc16` is one HLS kernel. It runs a **command list** from HBM. One launch runs a maximum of 128 commands in
sequence. The commands read and write FP32 vectors in one activation buffer. A command can use the result of
an earlier command without the host.

| Opcode | Command | Function |
| --- | --- | --- |
| 1 | `GEMV` | Q4_0, Q8_0, Q5_K or Q6_K weights × the vector in the activation buffer. Optional epilog (refer to 2.2) |
| 2 | `RMSNORM` | norm for each row. Optional weight, optional L2 variant, rows with a stride, optional *deferred* norm (refer to 2.2) |
| 3, 5 | `ADD`, `MUL` | element-wise operations |
| 4 | `SILU_MUL` | SwiGLU `silu(g) * u`, or the sigmoid gate `sigmoid(g) * u` of the attention output |
| 6 | `UNARY` | SiLU, sigmoid, softplus, copy. Rows with a stride |
| 7 | `GDN` | one Gated DeltaNet step (Qwen3.5). Optional: q/k L2 norm, gates, gated output norm, output projection |
| 8 | `CONV` | causal convolution for one token (kernel width 4), optional SiLU |
| 9 | `STATE_LOAD` | starts to read the state of the next `GDN` step into the chip while the next GEMVs run |
| 10 | `ROPE` | rotary position embedding (NEOX pairs) of the query and key heads. The host supplies the cos/sin table |
| 11 | `TRACE` | records the clock cycle of each command start and each pass start and end (`GGML_XRT_TRACE=1`) |

`fpga/src/gemvqc16.h` specifies the command format. A launch with zero commands writes a *feature word*. The
feature word shows the GEMV formats and the fusions of the bitstream. The backend reads it at start. The backend
uses a fusion only if the bitstream has it.

### 2.2 Pass

A *pass* is one dataflow region with the quantizer, the 20 engines, the result writer and the four DeltaNet units.
The hardware has one instance of it. One pass can do more than one command:

- **GEMV groups.** A maximum of four sequential GEMVs with the same input run as one pass. The quantizer quantizes
  the input one time. The engines read the matrices one after the other.
- **Epilogs.** A GEMV can have a SwiGLU pair (`g` and `u` of the FFN), the residual `ADD` or the `CONV` of the
  DeltaNet input. The result writer applies it. The intermediate vector does not go to the memory.
- **Deferred norm.** A deferred `RMSNORM` keeps its row in the chip. The next GEMVs on this row normalize the
  values when they quantize them. The arithmetic is the same as for the stored result.
- **DeltaNet step.** The `GDN` step uses a copy of the state in URAM. `STATE_LOAD` reads the next state before
  the step. The kernel writes the old state back during later GEMV passes.
- **Output projection in the DeltaNet pass.** The GEMV on the step output (`ssm_out`, with its residual add) runs
  in the pass of the step. This GEMV reads its weights in blocks of 256 columns. It starts when the first heads
  come out of the step.

Each fused form gives the same values as the plain command sequence.

### 2.3 GEMV engines

- Row `r` of a matrix is in engine `r % 20`. All engines work on their rows at the same time. Each engine writes
  one result for each row to the activation buffer.
- An engine processes one weight word (64 bytes) for each clock: 4 blocks of 32 weights for Q4_0, 2 blocks for Q8_0.
- The block dot products use exact integer arithmetic. For Q4_0 and Q8_0, the engine multiplies the two FP16 block
  scales (weight and activation) exactly on the significands. It adds the product to an 80-bit fixed-point
  accumulator. At the end of the row, it rounds to FP32 one time. The result does not depend on the sequence of the
  additions.
- The quantizer quantizes the activation vector as ggml does for the related weight type. It uses `Q8_0` for Q4_0
  and Q8_0 weights, and `Q8_K` for Q5_K and Q6_K weights. All engines get the same quantized vector.
- K-quants use super-blocks of 256 weights: one metadata word with the scales, then the quant words.

### 2.4 Weight layout (`gemvqw_layout.h`)

The host divides each GGUF row into groups. Each group has one 64-byte word with the FP16 block scales and then the
quant words. K-quants have one metadata word for each super-block. For Q6_K, three super-blocks share one metadata
word, and each super-block adds one word with the high bits.

Only the arrangement changes. The scale values and the quant values keep their original bits. The backend makes the
layout with `gemvqw_layout.h`. The kernel reads the same layout.

### 2.5 Memory map (`fpga/connectivity.cfg`)

| Port | HBM | Content |
| --- | --- | --- |
| `w0`..`w15` | even channels `2e` | weights of engine `e` (arena of 250 MiB) |
| `w16`..`w19` | channels 1, 3, 5, 7 | weights of engines 16–19 (arena of 250 MiB) |
| `act` | channel 17 | activation buffer: inputs, results, resident FP32 weights, convolution states (64 MiB) |
| `act_rd` | channel 17 | second read port on the same buffer, for the convolution epilog while the pass writes `act` |
| `cmds` | channel 17 | command list (4 KiB) |
| `st0`..`st3` | channels 19, 21, 23, 25 | DeltaNet states, 512-byte rows in sequence on the four ports |

With 16 engines (`GEMVQC16_ENGINES` 16), each engine port uses a channel pair `[2e:2e+1]` with an arena of
384 MiB. The backend reads the number of ports and the bank size from the bitstream.

## 3. ggml backend (`llama.cpp/overlay/ggml/src/ggml-xrt`)

- **Supported operations.** The backend accepts these nodes:
  - `MUL_MAT` with one token and a supported weight format
  - `RMS_NORM` and `L2_NORM` (with the next weight `MUL`), vector `ADD` and `MUL`
  - SwiGLU (`GLU`), SiLU, sigmoid and softplus (`UNARY`)
  - the causal convolution (`CONCAT` + `SSM_CONV`), `GATED_DELTA_NET`, `ROPE`, `CONT`
  - the `GET_ROWS`, `CPY` and `SCALE` nodes that move the recurrent state.

  The ggml scheduler gives all other nodes to the CPU backend.
- **Command lists.** The backend makes one list from each sequence of accepted nodes. It copies inputs from host
  memory to the input region of the activation buffer. The results stay on the card. The backend copies a result
  back only if a later CPU split or a graph output reads it.
- **Fusions.** The backend maps node patterns to the fused commands of section 2.2. `GGML_XRT_FEATURE_MASK` hides
  fusion bits of the kernel for diagnosis.
- **Plan cache.** The backend stores the analysis of a split (fusions, skipped nodes, use counts) under a
  fingerprint of the split structure. For the next token, it uses the stored analysis again. It makes only the
  commands again.
- **Weights.** The backend repacks and uploads a tensor when the graph uses it for the first time. There is no disk
  cache. Q4_0 weights stay in the CPU repack buffer of llama.cpp (AVX2 layout) for the prompt kernels of the CPU.
  The backend restores the GGUF rows from this buffer before the upload. The conversion has no loss.
- **Output head.** The output head has 248,320 rows. It goes to the card only if it fits into the arenas next to
  the layers. The backend calculates this one time from the sizes of all offloaded matrices and shows the result.
- **Recurrent state.** A special buffer type (patch `0004`) keeps the DeltaNet states and the convolution states
  on the card only. The CPU cannot access them: the address range has no access rights.
- **Graph order.** Patch `0006` puts all input projections of a layer next to each other. They then go into one list.
- **Kernel start.** If the scheduler of the driver is off (`ert_disable`), the backend writes the control registers
  of the kernel directly. This takes 6 µs. The start through XRT takes 0.2 ms.
- **Verification.** `GGML_XRT_VERIFY_CALLS` calculates the checked operations again with the reference
  implementation in `gemvq_ref.h`. It compares the results with the results of the card.

## 4. Time for each token

Measured with `GGML_XRT_TIMING`: 72 tokens minus 8 tokens (the weights are on the card). Reference bitstream (20
engines, kernel 94.8 MHz, HBM 347.9 MHz), direct start.

| Model | Time for each token (`llama-bench`) | Card busy | Weights for each token | Launches for each token | Card throughput |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B Q4_0 | 32.6 ms | 28.0 ms | 2.51 GB | 14 | 90 GB/s |
| Qwen3.5-4B Q8_0 | 47.3 ms | 44.4 ms | 4.47 GB | 10 | 101 GB/s |
| Qwen3.5-9B Q4_0 | 51.9 ms | 46.0 ms | 4.67 GB | 14 | 101 GB/s |

The other time (3–6 ms) is host work: the attention core of the 8 attention layers, the four Q4_1 matrices,
copies and sampling.

The 20 engines read a maximum of 20 × 64 B × 94.8 MHz = 121 GB/s. The cycle trace (`GGML_XRT_TRACE=1`) shows the
GEMV passes at 96 % of this value. The other card time has these causes:

- the DeltaNet steps: for 4B Q4_0, 5 ms for each token. The `ssm_out` stream hides 2 ms of this time.
- the norms
- the start and the end of each pass.

## 5. Limits of the design

- **Area.** The routed kernel uses 64 % of the LUTs and 78 % of the DSPs of the user budget. A variant with 32
  engines did not route. The LUTs limit the routing more than the DSPs.
- **Bandwidth.** The HBM can supply 2.5 times the bandwidth that the 20 engine ports use. More engines do not fit.
- **Compute.** For Q4_0, the engines are almost at their compute limit. The peak is 20 × 128 products × 94.8 MHz
  = 243 GMAC/s. The 4B model needs 4.4 GMAC for each token. At the streaming limit, this is 88 % of the multipliers.

These changes can increase the speed:

- a higher clock
- less work next to the weight stream
- one weight read for more than one token.

[ROADMAP.md](ROADMAP.md) gives the possible changes, their cost and their sequence.
