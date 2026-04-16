#pragma once
// HIP does not have a direct cuda_fp8.h equivalent
// Use hip_fp8.h if available, else provide stubs
#if __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>
#else
// FP8 types not available; kernels using FP8 will be disabled at runtime
#endif
