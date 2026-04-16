#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// CUDA Runtime API type aliases -> HIP
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString

// Memory management
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaFreeHost hipHostFree
#define cudaHostFree hipHostFree
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol

// Device management
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaDeviceSynchronize hipDeviceSynchronize

// Stream management
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking

// Event management
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventDefault hipEventDefault
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventInterprocess hipEventInterprocess

// IPC
#define cudaIpcMemHandle_t hipIpcMemHandle_t
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define cudaIpcEventHandle_t hipIpcEventHandle_t
#define cudaIpcGetEventHandle hipIpcGetEventHandle
#define cudaIpcOpenEventHandle hipIpcOpenEventHandle

// Error codes
#define cudaGetLastError hipGetLastError
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorCudartUnloading hipErrorDeinitialized
#define cudaErrorContextIsDestroyed hipErrorContextIsDestroyed
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaErrorInitializationError hipErrorNotInitialized
#define cudaErrorLaunchFailure hipErrorLaunchFailure

// Pointer attributes
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeDevice hipMemoryTypeDevice

// Memory type
#define cudaMemoryTypeHost hipMemoryTypeHost

// Data types (CUDA library types -> HIP)
#include <hip/library_types.h>
#define cudaDataType_t hipDataType
#define cudaDataType hipDataType
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F
#define CUDA_R_8I HIP_R_8I
#define CUDA_R_8U HIP_R_8U
#define CUDA_R_16I HIP_R_16I
#define CUDA_R_32I HIP_R_32I
#define CUDA_R_64I HIP_R_64I
#define CUDA_R_8F_E4M3 HIP_R_8F_E4M3_FNUZ
#define CUDA_C_16F HIP_C_16F
#define CUDA_C_32F HIP_C_32F
#define CUDA_C_64F HIP_C_64F

// Stream priority
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority

// Managed / Unified memory
#define cudaMallocManaged hipMallocManaged
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemAdvise hipMemAdvise

// Host pointer
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostAllocDefault hipHostMallocDefault

// Launch / Occupancy
#define cudaLaunchKernel hipLaunchKernel
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// IPC handle size
#define CUDA_IPC_HANDLE_SIZE HIP_IPC_HANDLE_SIZE

// Stream/device misc
#define cudaStreamQuery hipStreamQuery
#define cudaDeviceReset hipDeviceReset

// Note: HIP natively provides __shfl_*_sync and __syncwarp intrinsics
