# Roadmap: faster decode and prompt

Status 2026-11-07. This document gives the possible changes, their cost and their sequence. The values of open
items are estimates. The measured values come from [DESIGN.md](DESIGN.md#4-time-for-each-token).

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

1. **Bandwidth.** 20 engines × 64 bytes × 94.8 MHz = 121 GB/s. The HBM can supply 316 GB/s. 31 of the 32 HBM ports
   are in use.
2. **Bandwidth and compute are coupled.** A 64-byte word holds 128 Q4_0 values. An engine calculates 128 products
   for each cycle. For 20 engines, this is 243 GMAC/s. Qwen3.5-4B Q4_0 needs 4.4 GMAC for each token. At the
   streaming limit, 88 % of the multipliers are busy. **For Q4_0, the kernel is almost at its compute limit and at
   its bandwidth limit.** For this reason, more bandwidth needs more multipliers.
3. **Area.** The kernel uses 78 % of the DSPs and 64 % of the LUTs. The HBM-side SLR is the limit for the routing.
   The platform HBM interconnect does not meet 450 MHz.
4. **Clock.** The kernel runs at 94.8 MHz. The worst paths are in the DeltaNet stages and in the command loop
   (17–23 logic levels). Earlier builds at 150 MHz failed three times. At 150 MHz, each port needs 9.6 GB/s. A
   single HBM channel supplies 8–9 GB/s.
5. **Prompt.** The prompt is compute-bound. 512 tokens × 4.4 GMAC = 2.25 TMAC. The host CPU calculates
   280 GMAC/s, the engines a maximum of 243 GMAC/s. **With the current multipliers, the card cannot be faster than
   a CPU with 16 cores for the prompt.**

## 3. Open changes for decode

### 3.1 Less card time next to the weight stream

| ID | Change | Estimated gain (4B Q4_0) | Cost |
| --- | --- | --- | --- |
| K1 | More timing margin in HLS (pipeline stages in the DeltaNet units and the command loop) | +5 % (clock 100 MHz) | one link. A test build without C5 did not route: 2 % more LUTs, congestion level 7 |
| A5 | DeltaNet units with 16 lanes | −2 to −3 ms for each token | +560 DSPs. Possible only after C5 |
| A6 | Activations in the chip (URAM) for norms and element-wise passes | −1 to −2 ms for each token | URAM is 70 % free |
| A4 | `ssm_alpha`, `ssm_beta` and K/V projections in the qkv group | less than 0.5 ms | small |

### 3.2 Less work outside the card

The host needs 3–6 ms for each token (refer to [DESIGN.md](DESIGN.md#4-time-for-each-token)).

- **B1 Attention on the card.** This change puts the F16 KV cache into HBM on free ports. New opcodes do KV append,
  scores with softmax, and softmax × V. The change removes up to 8 of the 14 launches and the CPU attention. The
  estimated gain is 3–5 ms for each token. The numerics must follow the F16 path of the CPU. It is possible only
  as a separate vector kernel in the second SLR (C5).
- **B2 One launch for each token.** B1 makes this possible. Gain: 1–2 ms of synchronization.
- **B3 Q4_1 matrices on the card.** The four Q4_1 `ffn_down` tensors need +84 k LUT. The gain is 1 ms. This
  change stays off while the LUTs are full.

### 3.3 Higher bandwidth limit

- **C1 HBM bandwidth probe.** One link measures if one port with a channel pair supplies 12.8 GB/s. C2 needs this
  value.
- **C2 Two rows for each engine.** A two-row engine gives two times the bandwidth of an engine. It needs 194 DSPs in
  place of 96. The HLS engine is bit-exact. This change is possible only after C5.
- **C5 More area.** These changes make area free:
  - lossless weight metadata from the host (FP16 scales as mantissa and exponent, exact K-quant scale products)
  - one integer core for all formats (−30 to −50 % DSP for each engine)
  - one shared floating-point unit for exp, div, sqrt and log
  - two kernels: an engine kernel in the HBM-side SLR and a vector kernel in the other SLR, with AXI streams.

  The target is 900–1000 fewer DSPs outside the engines. The known candidates give 400–600 DSPs.

### 3.4 One weight read for more than one token

A batch path in the engines (P2) reads a weight word one time and multiplies it with M input vectors. With P2, two
uses are possible:

- **Speculative decode.** The card checks k draft tokens in one weight pass. A test with the available MTP model
  gave an acceptance of 4–7 % only. The recurrent state must go back to the last accepted token.
- **More sequences.** The card decodes more than one sequence in one pass. For Q8_0, two sequences have no
  compute cost. For Q4_0, the multiplier limit (section 2, item 2) limits the gain.

## 4. Changes for the prompt

| ID | Change | Status |
| --- | --- | --- |
| P0 | Q4_0 weights stay in the CPU repack buffer. The backend restores the GGUF rows for the upload | completed. pp512 on the card host: 62.6 / 58.3 / 34.8 tokens/s (CPU only 63.7 / 55.4 / 35.2) |
| P1 | Measure the prompt time for each part (GEMM, attention, DeltaNet chunk, other) for 128 and 512 tokens | open. Decides about P2 |
| P2 | Batch path in the engines: one weight word for M input vectors | open. Compute limit 47 tokens/s for 4B. Frees the CPU for the attention |
| P3 | Hybrid prompt: the CPU and the card each calculate a part of the rows of each matrix | open. On paper 63 + 47 tokens/s for 4B Q4_0. Synchronization for each matrix |
| P4 | Faster host CPU for long prompts | Ryzen 9 7950X: 136–210 tokens/s pp512 for 4B Q4_0 (CPU only) |

## 5. Sequence

| Step | Content | 4B Q4_0 | 4B Q8_0 | 9B Q4_0 |
| --- | --- | --- | --- | --- |
| current | 20 engines, 94.8 MHz (measured) | 30.7 | 21.1 | 19.3 |
| 1 area | C5 area reduction, target ≤ 3300 DSPs | as current | as current | as current |
| 2 clock | K1 | ≈ 32 | ≈ 22 | ≈ 20 |
| 3 split | engine kernel and vector kernel, A5 | ≈ 35 | ≈ 23 | ≈ 21 |
| 4 one launch | B1 attention on the card, B2 | ≈ 38–40 | ≈ 24–25 | ≈ 23–24 |
| 5 batch | P2 batch path, then speculative decode or more sequences | – | – | – |

The estimates use this equation: token time = streaming time + other card time + host time. Each step changes
one term.

Reasons for this sequence:

1. C5 gives no gain alone. But each later step needs its area. A K1 build without C5 did not route.
2. K1 needs one link and no change of function.
3. Steps 3 and 4 change the kernel count. They need the area of step 1.
4. P2 needs the multipliers that C2 or C5 makes free.

## 6. Risks and stop criteria

- If C1 shows less than 12.8 GB/s for each port, stop C2. The bandwidth limit then stays at the value of the
  current engine count.
- If C5 does not free 900 DSPs, do A5 or C2, not the two. Do B1 only as a separate kernel.
- If the integer core (C5) cannot show the scale products exactly, do not use an approximation. The design
  calculates the original model. Use only the host metadata and the shared floating-point unit.
- After each change in the dataflow region, run the RTL co-simulation of the DeltaNet sequence. Some errors were
  visible only in the RTL co-simulation.
- B1 numerics: the CPU calculates Q and the V accumulator in F16. The card must use the same path. If it does not,
  the logit difference becomes the acceptance criterion.
- More area can decrease the clock. After each link, compare the GEMV count and the launch count for each token in
  the cycle trace with the earlier state. A matrix that goes to the CPU without a message is almost not visible in
  tokens/s.
