# SPDX-License-Identifier: Apache-2.0
# More pipeline stages for the deep floating-point operators of gemvqc16. HLS picks the shortest latency that
# fits its estimate; after routing, such operators set the clock (an earlier build: fexp with latency 4, WNS
# -0.399 ns at 100 MHz, 55 % of the path routing). The Floating-Point IP computes the same results at any latency,
# and none of these operators sits in a floating-point recurrence, so only pipeline depths grow.
# Used by scripts/build_xclbin.sh (v++ -c --hls.pre_tcl).
config_op fexp -latency 8
config_op flog -latency 8
config_op fdiv -latency 12
config_op fsqrt -latency 12
