# Roadmap: faster decoding and prompt processing without raising the clock

Status 2026-10-18. Nothing in this document is implemented; the numbers are estimates derived from the measurements
in [DESIGN.md](DESIGN.md#where-the-time-goes). It records which levers exist, what they cost and in which order they
make sense, so the work can be picked up later.

## Where we stand

| Model | tokens/s | ms per token | card busy | streaming floor at 102.4 GB/s | card overhead | outside the card |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B Q4_0 | 24.2 | 41.4 | 32.3 | 24.5 | ≈ 8 ms | ≈ 9 ms |
| Qwen3.5-4B Q8_0 | 18.3 | 54.6 | 49.4 | 43.7 | ≈ 6 ms | ≈ 5 ms |
| Qwen3.5-9B Q4_0 | 15.9 | 63.0 | 53.0 | 45.6 | ≈ 7 ms | ≈ 10 ms |

"Outside the card" is list building, synchronisation and launches in the backend (4B Q4_0: 5.6 ms) plus CPU work
(attention core of the 8 attention layers, four Q4_1 matrices, sampling; 3.5 ms). Prompt processing runs entirely
on the CPU (4B Q4_0: 63 tokens/s CPU-only, 51 with the U50 binary, see the README's limitations).

## What limits the design

1. **Data path ceiling.** 16 engines × 64 bytes × 100 MHz = 102.4 GB/s; the HBM could deliver about 316 GB/s and
   11 of the 32 ports are unused.
2. **Bandwidth and compute are coupled.** A 64-byte word holds 128 Q4_0 values and the engine computes 128 products
   per cycle: 2048 MAC per cycle, 205 GMAC/s for 16 engines. Qwen3.5-4B Q4_0 needs about 4.4 GMAC per token, so
   at the 24.5 ms floor the multipliers are about 88 % busy. **For Q4_0 the kernel is almost compute-bound as
   well as bandwidth-bound.** Raising the ceiling always means more multipliers, which is why a two-row engine
   needs twice the DSPs. For Q8_0 (64 values per word) half of the multipliers idle.
3. **Area costs clock.** 3983 DSPs (67 %) just meet 100 MHz; 4759 DSPs ended at 79.5 MHz. The HBM-side SLR is 73 %
   DSP / 68 % LUT full and the platform's HBM interconnect no longer meets its 450 MHz.
4. **Clock alone buys little.** 150 MHz failed three times. At 150 MHz each port would need 9.6 GB/s, a single HBM
   pseudo channel delivers about 8–9 GB/s, and a matrix usually sits in one channel of the pair. Clock is the last
   step, after area and structure.
5. **Prompt processing is compute-bound.** 512 prompt tokens × 4.4 GMAC = 2.25 TMAC; the host CPU reaches about
   280 GMAC/s, the engines 205 GMAC/s peak. **With today's multipliers the card cannot beat a 16-core CPU at
   prompt processing,** even if every weight word were read only once (about 47 tokens/s versus 63).

## Levers for decoding

**A. Remove the overhead above the streaming floor** (up to −24 % card time for 4B Q4_0, at 100 MHz and 16 engines):

- A1 *On-card trace first*: counters per command (waiting for input, waiting for weights, output blocked, DeltaNet
  busy, idle between commands), read back through the activation buffer. Today only host-side timing exists; the
  figures below are guesses until this exists. Costs a few hundred LUTs.
- A2 *DeltaNet in parallel with the `ssm_out` GEMV*: with 8 lanes a DeltaNet step takes 16,384 cycles per layer, the
  z-GEMV hides about 5,600, leaving about 2.6 ms per token. A column-blocked GEMV with on-chip row accumulators
  (2560 rows, 160 per engine) can start as soon as the first head is done. Gain about 2.6 ms (4B Q4_0 → about
  25.8 tokens/s). Needs a column-wise weight order (host repack, bit-exact) and RTL co-simulation.
- A3 *Activations on chip*: norms and element-wise passes still go through the single activation port in HBM.
  Keep short vectors in URAM (about 70 % free). Estimated 1–2 ms per token.
- A4 *Small matrices*: `ssm_alpha`/`ssm_beta` (32 rows) and the K/V projections (1024 rows) barely fill the
  engines; group them with the qkv GEMV. Small gain (< 0.5 ms), cheap.
- A5 *DeltaNet back to 16 lanes* once area is free (−2 ms, +560 DSPs).

**B. Shrink the work outside the card** (4B Q4_0: about 9 ms = 22 % of the token time):

- B1 *Attention of the 8 attention layers on the card*: F16 KV cache in HBM on free ports, opcodes for KV append,
  scores + softmax, softmax·V. Removes up to 8 of the 14 launches and the CPU attention, estimated 3–5 ms per
  token. Only sensible as a separate vector kernel in the second SLR (C5); numerics must follow the CPU's F16 path.
- B2 *One launch per token* follows from B1; saves the remaining synchronisation (1–2 ms).
- B3 *Q4_1 matrices*: +84k LUT on the card, stays off; about 1 ms of CPU time.

**C. Raise the ceiling without more clock:**

- C1 *HBM bandwidth probe first* (one link run): does one port with both pseudo channels of its pair deliver
  12.8 GB/s? Without this measurement C2–C4 remain theory.
- C2 *Two rows per engine for 8 of the 16 engines* (+50 % ceiling). The two-row engine is verified bit-exact in
  HLS and costs 194 instead of 96 DSPs per engine; converting all 16 would need 97 % of the DSPs and does not fit.
  Eight wide engines: about +780 DSPs, affordable only after the area reduction (C5). Floors drop to 16.3 ms (4B
  Q4_0) and 29.1 ms (Q8_0); realistic 32–34 and 26–28 tokens/s.
- C3 *Two rows only for Q4_0/Q8_0* (no K-quant decoders in the wide engines; the Q6_K head and Q5_K `ssm_out` run
  at half rate).
- C4 *20 engines* (+25 %, 31 of 32 ports, about +500 DSPs; compiles and passes C simulation; conflicts with the
  state ports and the full HBM-side SLR).
- C5 *Make room* (prerequisite for A5, B1, C2–C4): precompute weight metadata on the host losslessly (FP16 scales
  as mantissa/exponent, exact K-quant scale products), one integer core for all formats (−30–50 % DSP per engine),
  one shared FP vector service (one exp, div, sqrt, log instead of a copy per fusion), an integer quantizer, and a
  split into an engine kernel (HBM-side SLR) and a vector kernel (other SLR) connected by AXI streams. Target: about
  900–1000 DSPs less outside the engines; the known candidates give 400–600, the rest must come from the integer
  core. The split also relieves the HBM interconnect so it can meet 450 MHz again.

**D. More work per weight word read:** speculative decoding was dropped (4–7 % acceptance with the available MTP
model, recurrent state must be rolled back). With a batch path (P2 below) verifying k draft tokens in one weight
pass becomes possible, and several sequences could decode in one pass. For Q4_0 the multiplier limit (point 2)
caps the benefit; for Q8_0 two sequences are free.

## Levers for prompt processing

- P0 *Fix the regression* (software only, first): keep an AVX2-repacked CPU copy of the offloaded Q4_0 layers after
  the card has uploaded the original layout (+2.5 GB RAM for 4B, +4.7 GB for 9B). Expected: pp512 4B Q4_0 51 → about
  63 tokens/s, 9B 27 → about 35, i.e. back to CPU level. Check with `llama-bench -p 512` against `-dev none`;
  decoding must not change.
- P1 *Measure the prompt breakdown* (GEMM, attention, DeltaNet chunk, rest) for 128 and 512 tokens, plus
  time-to-first-token for chat-length prompts. Decides whether P2 is worth it.
- P2 *Batch path in the engines*: hold a weight word for M cycles and multiply it with M input vectors from M local
  copies; row accumulators multiply by M. Weights are read once per prompt chunk; throughput becomes compute-bound
  at about 47 tokens/s for 4B (Q4_0 and Q8_0). Not a win against the CPU on its own, but it frees the CPU for the
  prompt's attention in parallel and is the basis for D. With C2 about 70 tokens/s. Decide after P1 and C1/C2.
- P3 *Hybrid prompt* (CPU and card split the rows of each matrix): on paper 63 + 47 → about 100 tokens/s for 4B
  Q4_0, but synchronisation per matrix and duplicated weights; evaluate with P1 numbers.
- P4 For long prompts a faster host CPU is the cheapest step (Ryzen 9 7950X: 136–210 tokens/s pp512 for 4B Q4_0);
  the card keeps its decoding gain.

## Order

| Phase | Content | 4B Q4_0 | 4B Q8_0 | 9B Q4_0 |
| --- | --- | --- | --- | --- |
| today | reference bitstream | 24.2 | 18.3 | 15.9 |
| 0 measure | A1 trace, C1 bandwidth probe, P0, P1 | – | – | – |
| 1 overhead | A2, A3, A4 at 100 MHz and 16 engines | ≈ 27–29 | ≈ 19.5–20 | ≈ 17.5–18 |
| 2 room | C5 area reduction; target ≤ 3300 DSPs, HBM domain meets 450 MHz | as phase 1 | as phase 1 | as phase 1 |
| 3 split | engine kernel / vector kernel per SLR; A5 | ≈ 29–30 | ≈ 20.5 | ≈ 18.5 |
| 4 ceiling | C2 eight two-row engines (if C1 ≥ 12.8 GB/s; else C4 or C3) | ≈ 32–34 | ≈ 26–28 | ≈ 22–24 |
| 5 one launch | B1 attention on the card, B2 | ≈ 36–38 | ≈ 28–30 | ≈ 24–26 |
| 6 batch | P2 batch path (prompt about 70 tokens/s with phase 4), D | – | – | – |
| last | clock 110–125 MHz only with the margin from phases 2–3; ≥ 150 MHz only with channel interleaving | | | |

Arithmetic behind the estimates: token time = streaming floor + card overhead + time outside the card. Phase 1
takes 3–4 ms off the overhead (4B Q4_0), phase 4 cuts the floor by a third, phase 5 takes 5–6 ms off the 9 ms outside
the card. Anything beyond that is not a serious estimate before A1 and C1 have been measured.

Why this order: phase 0 is almost free and decides where the 8 ms of overhead really are and whether the HBM side
can carry a higher ceiling at all. Phase 1 is the safest gain and stays within 16 engines and 100 MHz. Phase 2
brings no token gain by itself but is the condition for everything that needs area, and every change in it is
bit-exact against the test bench. Phases 3 and 4 change the engine width and the kernel count. Prompt processing
runs alongside: P0 immediately, P1 to decide, P2 only once the multipliers from phase 4 exist.

## Risks and stop criteria

- C1 shows less than 12.8 GB/s per port: C2 is off; check C4 or C3, otherwise the ceiling stays at 102.4 GB/s and
  the gains come from A and B only.
- Phase 2 does not free about 900 DSPs: then A5 or C2, not both; B1 only as a separate kernel.
- The integer core (C5) cannot represent the scale products exactly: no approximation (the design computes the
  original model); then host metadata and the vector service alone.
- A2 complicates the dataflow region: run the DeltaNet RTL sequence test after every step; the `gdn_store` class of
  bug was visible only in RTL co-simulation.
- B1 numerics: the CPU computes Q and the V accumulator in F16; the card must take the same path or the logit
  deviation becomes the acceptance criterion.
