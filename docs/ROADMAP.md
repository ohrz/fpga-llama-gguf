# Roadmap: faster decode and prompt

Status 2026-11-21. This document gives the possible changes, their cost and their sequence. The values of open
items are estimates. The measured values come from [DESIGN.md](DESIGN.md#4-time-for-each-token).

The data in this document have three sources:

| Mark | Source |
| --- | --- |
| measured | a measurement on the card |
| HLS | a high-level synthesis report or a C++ model test without the card (2026-11-21) |
| reported | a value from a publication. We did not measure it |

## 1. Completed changes

| ID | Change | Result |
| --- | --- | --- |
| A1 | Cycle trace in the kernel (opcode `TRACE`) | Shows the cycles of each command type. GEMV passes are at 96 % of the ideal value |
| A2 | `ssm_out` GEMV in the DeltaNet pass | 5–8 % fewer kernel cycles than the same kernel without the fusion |
| A3 | Sigmoid gate of the attention output as one command | included in the A2 result |
| P0 | Prompt at CPU speed again | Q4_0 weights stay in the CPU repack buffer. pp512 with the card is at CPU level |
| C4 | 20 engines, one HBM channel for each engine | 30 % fewer kernel cycles for each token |
| – | Output head on the card with 250 MiB arenas | The 4B and 9B models fit. Without it, the CPU needs 8 ms for each token |
| – | Direct start as default | 12–15 % more tokens/s than the XRT start |

Two area changes (LUTRAM buffers, adders in LUTs) made the design unroutable. We removed them. On the U50, the LUTs
limit the routing more than the DSPs.

A2 and A3 alone gave no net gain on the card. The `z` projection moved out of the DeltaNet pass, and the clock
decreased by 1 %. The table in the [README](../README.md#performance-history) shows the values of each step.

## 2. Limits of the current design

### 2.1 Time model

For Qwen3.5-4B Q4_0, the token time has four parts (measured):

| Part | Content | Time |
| --- | --- | --- |
| Stream | 2.51 GB of weights at 121 GB/s and 96 % use | 21.6 ms |
| DeltaNet | the part of the DeltaNet steps that the `ssm_out` stream does not hide | 2.9 ms |
| Other card time | norms, start and end of each pass, small commands | 3.5 ms |
| Host | attention core, four Q4_1 matrices, copies, sampler | 4.6 ms |
| Sum | 30.7 tokens/s | 32.6 ms |

The weight stream is two thirds of the token time. The stream alone permits a maximum of 46 tokens/s. For a
larger gain, the bandwidth must increase.

### 2.2 Limits

1. **Bandwidth.** 20 engines × 64 bytes × 94.8 MHz = 121 GB/s. The HBM can supply 316 GB/s. 31 of the 32 HBM ports
   are in use.
2. **Bandwidth for each port.** An engine port moves 6.07 GB/s. Published measurements report 12.9 to 13.2 GB/s for
   one HBM channel ([Choi et al.](https://arxiv.org/abs/2010.06075)). We did not measure this value. If it is
   correct, the engine ports use less than half of their channels.
3. **Bandwidth and compute change together.** A 64-byte word holds 128 Q4_0 values. An engine calculates 128 products
   for each cycle. For 20 engines, this is 243 GMAC/s. Qwen3.5-4B Q4_0 needs 4.4 GMAC for each token. At the
   streaming limit, 88 % of the multipliers are busy. For this reason, more bandwidth needs more multipliers or a
   higher engine clock.
4. **Area.** The kernel uses 78 % of the DSPs and 64 % of the LUTs. The HBM-side SLR is the limit for the routing.
   A test build with 2 % more LUTs did not route.
5. **Clock.** The kernel runs at 94.8 MHz. The worst paths are in the DeltaNet stages and in the command loop
   (17–23 logic levels). They are not in the engines (refer to section 3.3). The platform HBM interconnect does not
   meet 450 MHz.
6. **Prompt.** The prompt is compute-bound. 512 tokens × 4.4 GMAC = 2.25 TMAC. The host CPU calculates
   280 GMAC/s, the engines a maximum of 243 GMAC/s. With the current multipliers, the card cannot be faster than
   a CPU with 16 cores for the prompt.

### 2.3 Content of the weight stream

Bytes for each token of `Qwen3.5-4B-Q4_0.gguf`:

| Group | Format | MB | Part |
| --- | --- | --- | --- |
| FFN matrices | Q4_0 | 1221 | 47.5 % |
| Output head | Q6_K | 522 | 20.3 % |
| DeltaNet input projections | Q4_0 | 425 | 16.5 % |
| `ssm_out` | Q5_K | 173 | 6.7 % |
| Attention projections | Q4_0 | 165 | 6.4 % |
| Four `ffn_down` matrices (CPU) | Q4_1 | 59 | 2.3 % |
| `ssm_alpha`, `ssm_beta` | Q8_0 | 4 | 0.2 % |

The K-quant matrices are 27 % of the stream. A change that makes only Q4_0 two times faster gives a maximum of
1.57× for the stream.

## 3. Open changes for decode

### 3.1 Less card time next to the weight stream

| ID | Change | Estimated gain (4B Q4_0) | Cost |
| --- | --- | --- | --- |
| K1 | More timing margin in HLS (pipeline stages in the DeltaNet units and the command loop) | +5 % (clock 100 MHz) | one link. A test build without C5 did not route: 2 % more LUTs, congestion level 7 |
| A7 | DeltaNet step in two phases (refer to the text below the table) | −2 ms for each token | +356 DSPs, +23,800 LUTs (HLS estimate). The kernel source has it as a build switch. Possible only after C5 |
| A5 | DeltaNet units with 16 lanes | −2 ms for each token | +560 DSPs and +24,900 LUTs (HLS estimate). A7 gives the same step time with fewer DSPs |
| A6 | Activations in the chip (URAM) for norms and element-wise passes | −1 to −2 ms for each token | URAM is 70 % free |
| A4 | `ssm_alpha`, `ssm_beta` and K/V projections in the qkv group | less than 0.5 ms | small |

**A7 in detail.** The step calculates the output from the new state `S' = x + k·δ`. The two-phase form calculates
the output as `(x·q) + δ·(k·q)`. The output then does not wait for the new state. A C++ study with 1.6 million
values gave the same accuracy for the two forms: mean relative error 1.18e-07 (current) and 1.14e-07 (two phases).

The kernel source has this form as a build switch (`GEMVQC_GDN_TWO_PHASE` in `fpga/src/gemvqc16.h`, default 0).
The default kernel does not change: its HLS estimate is the same as before.

- **Output path.** Stage A of each DeltaNet unit uses 16 lanes. It calculates `x`, `δ` and the output. Before the
  rows of a head, the words of `k` go through the same lanes. This gives `k·q` without more multipliers.
- **State update.** Stage B keeps the decayed rows and `δ`. It calculates `S' = x + k·δ` with 8 lanes when it writes
  the rows to the HBM. This occurs in the passes after the step, next to the engines. The write-back of a step is
  complete before the kernel loads the next step.
- **Commands.** The command list is the same. The backend does not change.

Results:

| Test | Result |
| --- | --- |
| C simulation of the kernel test bench | 193 of 193 cases agree |
| State after each of the 84 launches of the test bench | the same bits as with the default kernel |
| Output values | some values are different in the last bits. All tolerance checks agree |
| RTL co-simulation of the DeltaNet sequence (20 engines) | 27 of 27 cases agree |
| Emulator, llama.cpp `b9972` and `b12602` | 742 and 754 checked operations agree. The text is the same |
| RTL co-simulation of a test list: two layers with 16 heads, commands with fusions (16 engines) | The results agree. 46,345 cycles. The default kernel needs 52,760 cycles |
| The same test list, commands without fusions | 78,029 cycles. The default kernel needs 86,273 cycles |
| Pass of the step in this test list (16 heads) | 5,967 cycles. The default kernel needs 10,089 cycles |
| HLS estimate | +356 DSPs, +23,800 LUTs, +33,700 FFs, +8 BRAMs. The clock estimate is the same |

One head needs 264 cycles in the step. The default kernel needs 512 cycles. The 4B model has 24 DeltaNet layers
with 32 heads. Thus the estimated gain is 190,000 cycles or 2.0 ms for each token at 94.8 MHz. This is an estimate
from the cycle count. It is not a measurement on the card.

The state update needs two clocks for each word. The default kernel needs one clock to write a word. If the GEMV
passes after a step are too short, the kernel adds a pass for the remaining words. In the test list, this pass
adds 1,817 cycles after the last command. The GEMV passes of a 4B layer are much longer than those of the test list.

The first estimate of the cost (130 to 190 DSPs) was too low. The output path needs three multipliers for each
lane, not two. We did not build a bitstream with this switch. The design has no free area for it before C5.

### 3.2 Less work outside the card

The host needs 3–6 ms for each token (refer to [DESIGN.md](DESIGN.md#4-time-for-each-token)).

- **B1 Attention on the card.** This change puts the F16 KV cache into HBM on free ports. New opcodes do KV append,
  scores with softmax, and softmax × V. The change removes up to 8 of the 14 launches and the CPU attention. The
  estimated gain is 3–5 ms for each token. The numerics must follow the F16 path of the CPU. It is possible only
  as a separate vector kernel in the second SLR (C5).
- **B2 One launch for each token.** B1 makes this possible. Gain: 1–2 ms of synchronization.
- **B3 Q4_1 matrices on the card.** A Q4_1 weight is `d·q + m`. The added term is `m` × the block sum of the
  input. The quantizer can supply the block sum. The scale lanes of the engine can calculate the product. This form
  needs some hundred LUTs for each engine (estimate, no synthesis). The earlier form needed +84 k LUT. The gain is
  1 ms and four launches.

### 3.3 Higher bandwidth limit

- **C1 HBM bandwidth probe.** One link measures the throughput of one port on one channel at 256 bits. Changes
  C2 and C6 need 12.1 GB/s. **This is the most important open measurement.**
- **C2 Two rows for each engine.** A two-row engine reads 128 bytes for each clock. It needs an HBM clock of
  380 MHz or more.
- **C6 Engine kernel with its own clock.** The engines become a separate kernel with a higher clock. The DeltaNet
  units and the command loop keep their clock. Each clock step gives a gain: 125 MHz gives +32 % bandwidth.
- **C7 Two products for each DSP.** The two rows of a two-row engine use the same input value. Publications
  report that one DSP can then calculate the two products. We did not test this in HLS.
- **C5 More area.** These changes make area free:
  - lossless weight metadata from the host (FP16 scales as mantissa and exponent, exact K-quant scale products)
  - one integer core for all formats (−30 to −50 % DSP for each engine)
  - one shared floating-point unit for exp, div, sqrt and log
  - two kernels: an engine kernel in the HBM-side SLR and a vector kernel in the other SLR, with AXI streams.

  The target is 900–1000 fewer DSPs outside the engines. The known candidates give 400–600 DSPs.

  Status (kernel source, HLS estimate, no bitstream): the kernel has 758 fewer DSPs (4025 to 3267). The results
  have the same bits.
  - Each engine has 86 DSPs instead of 120 (−680 DSPs for 20 engines). One DSP multiplier calculates the sum of
    two 4-bit weight products. The K-quant scale products and the FP16 scale products use the same two multipliers
    in each lane.
  - The DeltaNet load and the norms share one unit for the sum of squares (−78 DSPs).
  - The LUT estimate increases by 72,000 (+9 %): about 3,800 for each engine. The routed LUT count is not
    known.
  - Not done: the host metadata, the shared unit for exp, div, sqrt and log, and the two kernels.

HLS results for one engine (`xcu50`, estimates before implementation):

| Engine variant | Rows for each clock | LUT | DSP |
| --- | --- | --- | --- |
| One row, all formats (current) | 1 | 18,122 | 96 |
| One row, Q4_0 and K-quants | 1 | 13,646 | 94 |
| One row, Q4_0 and Q8_0 | 1 | 12,202 | 74 |
| One row, Q4_0 only | 1 | 10,551 | 72 |
| Two rows, all formats | 2 | 33,610 | 194 |
| Two rows, Q4_0 and Q8_0 | 2 | 21,116 | 150 |
| One row, all formats, target 150 MHz | 1 | 18,194 | 96 |
| One row, all formats, target 200 MHz | 1 | 18,476 | 96 |
| Two rows, Q4_0 and Q8_0, target 200 MHz | 2 | 21,786 | 150 |

Conclusions:

- The K-quant path is 5.9 k LUT and 22 DSPs of each engine. The Q8_0 path is 4.5 k LUT.
- A two-row engine for Q4_0 and Q8_0 needs 17 % more LUTs and 56 % more DSPs than the current engine.
- With a one-row K-quant path, 20 two-row engines need approximately 1500 more DSPs (estimate). They do not fit
  into the current chip. C2 is possible only after C5.
- A higher clock target changes the engine area by less than 2 %. The engine is not the cause of the 94.8 MHz.

### 3.4 One weight read for more than one token

A batch path in the engines (P2) reads a weight word one time and multiplies it with M input vectors. With P2, two
uses are possible:

- **Speculative decode.** The card checks k draft tokens in one weight pass. A test with the available MTP model
  gave an acceptance of 4–7 % only. Newer llama.cpp versions have more draft types and can restore the recurrent
  state (refer to section 3.5).
- **More sequences.** The card decodes more than one sequence in one pass. For Q8_0, two sequences have no
  compute cost. For Q4_0, the multiplier limit (section 2.2, item 3) limits the gain.

### 3.5 Newer llama.cpp

The default version is llama.cpp release `b9972` (July 2026). All measured values use it. On 2026-11-21, the newest
release was `b12602`. The build script can use it: `LLAMA_CPP_VERSION=b12602 scripts/build_llama.sh`.

| ID | Item | Status |
| --- | --- | --- |
| U1 | Backend on `b12602` | completed in the emulator. All checked operations agree, and the text is the same as on the CPU |
| U2 | Fusions on the new graph | completed in the emulator. The same run has 174 graph calls with the two versions |
| U5 | New L2 norm formula in the DeltaNet command | completed in the kernel source and in the emulator. The card needs a new bitstream. Estimated gain: 0.3 ms for each token |
| U3 | Fused Qwen3.5 weights (`wqkv_z`, `ssm_beta_alpha_cat`) | open. Fewer and larger GEMV groups |
| U4 | Measurement on the card | completed (2026-11-21, current bitstream). All checked operations agree, and the text is the same as on the CPU. `b12602` is 0.43–0.49 ms slower for each token than `b9972` (4B Q4_0: 30.29 tokens/s, `b9972`: 30.69). The bitstream does not have U5, so the backend uses two RMSNORM commands for each DeltaNet layer |

Changes that were necessary for U1 and U2:

- one new parameter in a RoPE function call. The backend finds the llama.cpp version during the build
- a new graph-order patch for the new model builder (`llama.cpp/patches/b12602`)
- the `FILL` operation on the recurrent state buffer
- the `SCALE` operation after an `RMS_NORM`.

**Cause of the missing fusions.** llama.cpp `b12602` calculates the L2 norm of q and k as `RMS_NORM` and `SCALE`.
The backend did not accept this `SCALE`. The CPU calculated it. This divided each DeltaNet layer into five splits.
The convolution and the DeltaNet step were then in different splits, and the fusions between them were not possible.
The backend now sends the `RMS_NORM` and its `SCALE` to the card as one `RMSNORM` command with a constant weight.
All other fusions are the same as with `b9972`: the convolution epilog, the gates, the gated norm and `ssm_out`
in the DeltaNet pass.

**The L2 norm (U5).** The two versions use different formulas:

| Version | Formula | Current bitstream | Kernel source |
| --- | --- | --- | --- |
| `b9972` | `x / max(sqrt(sum), eps)` | part of the DeltaNet command | part of the DeltaNet command |
| `b12602` | `x / sqrt(sum + eps)` | `RMSNORM` command with a constant weight | part of the DeltaNet command (flag `GEMVQC_GDN_L2_SUM`) |

We tested the old formula in the DeltaNet command with the new graph. The emulator check failed in layer 1. The
smallest sum of squares of a head was 0.006, and the relative difference was 8e-5. Thus, with the current
bitstream, the backend uses the exact form. Each DeltaNet layer has two more `RMSNORM` commands (2048 values
each). The estimated cost is 0.3 ms for each token. This estimate uses the measured mean time of the norm commands
(approximately 6 µs).

The kernel source now has the new formula in the DeltaNet command. The two formulas use the same square root and
the same divider. A feature bit shows the backend that the kernel has the new flag. If the bit is not set, the
backend uses the `RMSNORM` commands. Thus the backend operates with the two bitstreams. The results:

- C simulation of the kernel: 193 of 193 cases agree. The new command gives the same bits as a DeltaNet step with
  q and k that the host normalized.
- Emulator with `b12602`: 754 checked operations agree. The 96 `RMSNORM` commands of the run are not necessary.
  The check compares the DeltaNet command with the `RMSNORM` commands that it replaces.
- Emulator with `b12602` and without the feature bit: 850 checked operations agree, as before.
- Emulator with `b9972`: 742 checked operations agree, as before.
- HLS estimate: +125 LUT and +400 FF, no more DSPs, the same clock estimate.

We did not build a bitstream with this kernel source. The measured bitstream does not have the new flag.

In `b12602`, the graph selects the output token before the last residual addition. In `b9972`, it selects the
output token after the last norm. Thus, with `b12602`, the card also calculates the end of the last layer of a
prompt. This is the reason for the different number of checked operations in the two versions.

### 3.6 Changes that we examined and do not use

| Change | Result |
| --- | --- |
| Lossless compression of the weight stream | The entropy of the Q4_0 values is 3.76 of 4 bits. The maximum gain is 1.06× |
| Calculation of a part of the output head only | No exact selection rule is known. The bounds are too large |
| Q8_0 as two 4-bit lanes | HLS: 600 more LUTs, the same DSPs |
| A mix of two-row and one-row engines | Slower than the current design, because K-quants are 27 % of the stream |

## 4. Changes for the prompt

| ID | Change | Status |
| --- | --- | --- |
| P0 | Q4_0 weights stay in the CPU repack buffer. The backend restores the GGUF rows for the upload | completed. pp512 on the card host: 62.6 / 58.3 / 34.8 tokens/s (CPU only 63.7 / 55.4 / 35.2) |
| P1 | Measure the prompt time for each part (GEMM, attention, DeltaNet chunk, other) for 128 and 512 tokens | open. Decides about P2 |
| P2 | Batch path in the engines: one weight word for M input vectors | open. Compute limit 47 tokens/s for 4B. Frees the CPU for the attention |
| P3 | Hybrid prompt: the CPU and the card each calculate a part of the rows of each matrix | open. On paper 63 + 47 tokens/s for 4B Q4_0. Synchronization for each matrix |
| P4 | Faster host CPU for long prompts | Ryzen 9 7950X: 136–210 tokens/s pp512 for 4B Q4_0 (CPU only) |

## 5. Sequence

| Step | Content | Token time 4B Q4_0 | Tokens/s 4B Q4_0 |
| --- | --- | --- | --- |
| current | 20 engines, 94.8 MHz (measured) | 32.6 ms | 30.7 |
| 1 | U4: measure llama.cpp `b12602` on the card (U2 is complete) | 33.0 ms (measured, +0.43 ms) | 30.3 |
| 2 | C1: measure the port bandwidth | as current | as current |
| 3 | C5: area reduction, target ≤ 3300 DSPs (kernel source: 3267 DSPs, HLS estimate) | as current | as current |
| 4 | A7 two-phase DeltaNet step, B3 Q4_1 on the card | ≈ 29.6 ms | ≈ 34 |
| 5 | Two kernels, then C2: two times the bandwidth for Q4_0 and Q8_0 | ≈ 21.6 ms | ≈ 46 |
| 5 (alternative) | Two kernels, then C6 at 190 MHz: two times the bandwidth for all formats | ≈ 18.6 ms | ≈ 54 |
| 6 | B1 attention on the card, B2 one launch (after step 5 with C2) | ≈ 19.5 ms | ≈ 51 |
| 7 | C2 and an engine clock of 125 MHz | ≈ 17–18 ms | ≈ 56–60 |
| limit | 252 GB/s for all formats, other parts as in step 6 | ≈ 16 ms | ≈ 62 |
| 8 | P2 batch path, then speculative decode or more sequences | – | – |

The estimates use the time model of section 2.1. Each step changes one part. The values for steps 4 to 7 are
estimates. They are not results. The estimates use these assumptions:

- A7 decreases the DeltaNet part from 2.9 ms to 0.9 ms.
- B3 moves 59 MB to the card (+0.5 ms stream) and removes 1.5 ms of host time.
- B1 and B2 decrease the host time to 1 ms.
- We assume that one HBM channel supplies 12.6 GB/s. In step 7, this value limits the Q4_0 stream. The K-quant
  stream gets the full gain of the clock.
- The other card time (3.5 ms) does not change. If it decreases to half, the limit is approximately 70 tokens/s.

C6 at 190 MHz gives a larger gain than C2, because it also makes the K-quant stream two times faster. It needs
almost no engine area. But it needs timing closure of the engine kernel at two times the current clock.

Reasons for this sequence:

1. Steps 1 and 2 need no change of the kernel function. Step 2 decides between C2, C6 and the current limit.
2. C5 gives no gain alone. But each later step needs its area. A K1 build without C5 did not route.
3. Step 4 decreases the card time next to the stream. A7 needs 356 DSPs and 23,800 LUTs (HLS estimate).
4. Step 5 gives the largest gain. It needs the area of step 3 and the result of step 2.
5. P2 needs multipliers that are not free in the current design.

## 6. Risks and stop criteria

- If C1 shows less than 12.1 GB/s for each port, stop C2. Use C6 to the measured port limit only.
- If the tool cannot give a different clock to each kernel on this platform, stop C6. Use C2.
- If C5 does not free 900 DSPs, do A7 or C2, not the two. Do B1 only as a separate kernel.
- If the integer core (C5) cannot show the scale products exactly, do not use an approximation. The design
  calculates the original model. Use only the host metadata and the shared floating-point unit.
- After each change in the dataflow region, run the RTL co-simulation of the DeltaNet sequence. Some errors were
  visible only in the RTL co-simulation.
- B1 numerics: the CPU calculates Q and the V accumulator in F16. The card must use the same path. If it does not,
  the logit difference becomes the acceptance criterion.
- More area can decrease the clock. After each link, compare the GEMV count and the launch count for each token in
  the cycle trace with the earlier state.
- A matrix that goes to the CPU without a message is almost not visible in tokens/s. The launch count shows it.
