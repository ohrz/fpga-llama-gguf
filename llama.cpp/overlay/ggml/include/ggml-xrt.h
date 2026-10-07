// SPDX-License-Identifier: Apache-2.0
#ifndef GGML_XRT_H
#define GGML_XRT_H

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Entry point that ggml-backend-reg.cpp calls to register the Alveo U50 XRT backend.
ggml_backend_reg_t ggml_backend_xrt_reg(void);

// Buffer type of the recurrent-state tensors (llama.cpp's r_l / s_l caches) of a gemvqc16 kernel with DeltaNet
// opcodes: the states live on the card. Returns NULL when the active kernel has no DeltaNet support.
ggml_backend_buffer_type_t ggml_backend_xrt_rs_buffer_type(ggml_backend_dev_t dev);

#ifdef __cplusplus
}
#endif

#endif // GGML_XRT_H
