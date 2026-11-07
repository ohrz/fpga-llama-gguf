# SPDX-License-Identifier: Apache-2.0
#
# ggml backend for the gemvqc16 command-list kernel on the AMD Alveo U50 (XRT).

# XRT headers and libraries: $XILINX_XRT, or /opt/xilinx/xrt.
set(XRT_PATH "$ENV{XILINX_XRT}")
if (NOT XRT_PATH)
    set(XRT_PATH "/opt/xilinx/xrt")
endif()
find_library(XRT_CORE_LIB NAMES xrt_coreutil HINTS "${XRT_PATH}/lib")
if (NOT XRT_CORE_LIB)
    message(FATAL_ERROR "ggml-xrt: libxrt_coreutil not found in ${XRT_PATH}/lib (set the XILINX_XRT environment variable)")
endif()
message(STATUS "ggml-xrt: XRT in ${XRT_PATH}")

# The host code shares the weight layout (gemvqw_layout.h) and the CPU reference (gemvq_ref.h) with the kernel
# sources: point this at fpga/src of this repository.
set(GGML_XRT_GEMVQ_INCLUDE "" CACHE PATH "Directory with the gemvqc16 kernel sources (fpga/src)")
if (NOT GGML_XRT_GEMVQ_INCLUDE)
    message(FATAL_ERROR "ggml-xrt: set -DGGML_XRT_GEMVQ_INCLUDE=<repository>/fpga/src")
endif()

ggml_add_backend_library(ggml-xrt
                         ggml-xrt.cpp)

target_include_directories(ggml-xrt PRIVATE "${XRT_PATH}/include" "${GGML_XRT_GEMVQ_INCLUDE}")
target_link_libraries(ggml-xrt PRIVATE "${XRT_CORE_LIB}" Threads::Threads)
target_compile_features(ggml-xrt PRIVATE cxx_std_17)

# Optional: run the kernel as its HLS C++ model on the host (GGML_XRT_EMULATE=1, no card needed). The model is
# compiled from the kernel sources against the Vitis HLS headers (ap_int.h, hls_stream.h).
set(GGML_XRT_EMULATE_INCLUDE "" CACHE PATH "Vitis HLS include directory (contains ap_int.h); enables GGML_XRT_EMULATE=1")
set(GGML_XRT_EMULATE_ENGINES "20" CACHE STRING "engines of the emulated gemvqc16 (20 or 16, as the kernel)")
if (GGML_XRT_EMULATE_INCLUDE)
    add_library(ggml-xrt-gemvqc16-model STATIC "${GGML_XRT_GEMVQ_INCLUDE}/gemvqc16.cpp")
    target_include_directories(ggml-xrt-gemvqc16-model PRIVATE "${GGML_XRT_GEMVQ_INCLUDE}" "${GGML_XRT_EMULATE_INCLUDE}")
    target_compile_options(ggml-xrt-gemvqc16-model PRIVATE -fno-strict-aliasing -w)
    target_compile_definitions(ggml-xrt-gemvqc16-model PRIVATE GEMVQC16_ENGINES=${GGML_XRT_EMULATE_ENGINES})
    set_target_properties(ggml-xrt-gemvqc16-model PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 14)
    target_link_libraries(ggml-xrt PRIVATE ggml-xrt-gemvqc16-model)
    target_compile_definitions(ggml-xrt PRIVATE GGML_XRT_GEMVQ_EMULATE GGML_XRT_EMULATE_ENGINES=${GGML_XRT_EMULATE_ENGINES})
    message(STATUS "ggml-xrt: gemvqc16 C++ model for emulation (${GGML_XRT_EMULATE_INCLUDE})")
endif()
