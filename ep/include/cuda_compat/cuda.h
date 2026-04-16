#pragma once
#ifndef CUDA_COMPAT_DRIVER_H
#define CUDA_COMPAT_DRIVER_H
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
typedef hipDeviceptr_t CUdeviceptr;
typedef hipError_t CUresult;
typedef hipCtx_t CUcontext;
typedef int CUdevice;
#define CUDA_SUCCESS hipSuccess
#define cuMemGetAddressRange hipMemGetAddressRange
#define cuCtxGetCurrent hipCtxGetCurrent
#define cuCtxSetCurrent hipCtxSetCurrent
#define cuDeviceGet hipDeviceGet
#define cuDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxRelease hipDevicePrimaryCtxRelease
#define cuGetErrorString hipDrvGetErrorString
#endif
