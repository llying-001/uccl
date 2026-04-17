"""
Phase 1.2: Test RDMA Proxy initialization in single-process multi-GPU mode.

Key risk validation:
- Can multiple Proxy instances (is_intranode=False) coexist in one process?
- Does ibv_reg_mr succeed for GPU memory on each device?
- Does allocate_rdma_buffer work per-GPU?
"""
import sys, os, ctypes, traceback
import numpy as np

EP_BUILD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EP_BUILD_DIR)

_hip = ctypes.CDLL("libamdhip64.so")

def hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")

def hip_set_device(dev):
    hip_check(_hip.hipSetDevice(ctypes.c_int(dev)), f"hipSetDevice({dev})")

def hip_get_device_count():
    c = ctypes.c_int()
    hip_check(_hip.hipGetDeviceCount(ctypes.byref(c)), "getDeviceCount")
    return c.value

def hip_malloc(nbytes):
    ptr = ctypes.c_void_p()
    hip_check(_hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), f"hipMalloc({nbytes})")
    return ptr.value

def hip_free(ptr):
    hip_check(_hip.hipFree(ctypes.c_void_p(ptr)), "hipFree")

def hip_memset(ptr, val, nbytes):
    hip_check(_hip.hipMemset(ctypes.c_void_p(ptr), ctypes.c_int(val), ctypes.c_size_t(nbytes)), "hipMemset")

def hip_device_synchronize():
    hip_check(_hip.hipDeviceSynchronize(), "sync")

def hip_device_enable_peer_access(peer):
    err = _hip.hipDeviceEnablePeerAccess(ctypes.c_int(peer), ctypes.c_int(0))
    if err != 0 and err != 704:
        hip_check(err, f"peerAccess({peer})")

def cleanup_shm():
    import glob as g
    for f in g.glob("/dev/shm/uccl_barrier_*"):
        try: os.remove(f)
        except: pass


def test_rdma_proxy_init():
    cleanup_shm()
    import ep

    num_gpus = min(hip_get_device_count(), 8)
    num_rdma_gpus = min(num_gpus, 4)  # Test with 4 GPUs for RDMA to save resources
    rdma_bytes = 1 << 24  # 16MB RDMA buffer per GPU
    nvl_bytes = 1 << 26   # 64MB NVL buffer per GPU
    num_proxy_threads = ep.get_num_proxy_threads()

    print(f"[RDMA-INIT] Testing RDMA Proxy init with {num_rdma_gpus} GPUs")
    print(f"[RDMA-INIT] rdma_bytes={rdma_bytes}, nvl_bytes={nvl_bytes}, proxy_threads={num_proxy_threads}")

    # Enable peer access
    for i in range(num_rdma_gpus):
        hip_set_device(i)
        for j in range(num_rdma_gpus):
            if i != j:
                hip_device_enable_peer_access(j)

    # Step 1: Test get_rdma_buffer per-GPU
    print("\n[RDMA-INIT] Step 1: get_rdma_buffer per-GPU...")
    rdma_bufs = []
    rdma_ptrs = []
    rdma_is_host = []

    def dlpack_data_ptr(capsule):
        """Extract raw data pointer from a DLPack PyCapsule (DLManagedTensor)."""
        PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
        PyCapsule_GetPointer.restype = ctypes.c_void_p
        PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        managed_ptr = PyCapsule_GetPointer(capsule, b"dltensor")
        if not managed_ptr:
            raise RuntimeError("PyCapsule_GetPointer returned NULL")
        data_ptr = ctypes.c_void_p.from_address(managed_ptr).value
        return data_ptr

    for gpu_id in range(num_rdma_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        try:
            buf, is_host = ep.get_rdma_buffer(rdma_bytes, gpu_id)
            ptr_val = dlpack_data_ptr(buf)
            rdma_bufs.append(buf)
            rdma_ptrs.append(ptr_val)
            rdma_is_host.append(is_host)
            print(f"  GPU {gpu_id}: rdma_buffer allocated, ptr=0x{ptr_val:x}, is_host={is_host}")
        except Exception as e:
            print(f"  GPU {gpu_id}: FAILED to allocate rdma_buffer: {e}")
            traceback.print_exc()
            return
    print("[RDMA-INIT] Step 1 PASSED!")

    # Step 2: Create Proxy instances with is_intranode=False (RDMA mode)
    print("\n[RDMA-INIT] Step 2: Create internode Proxy (is_intranode=False)...")
    all_proxies = []
    # Simulate 2-node 16-rank topology for RDMA path
    num_virtual_ranks = num_rdma_gpus * 2  # pretend 2 nodes
    for gpu_id in range(num_rdma_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        proxies = []
        for t in range(num_proxy_threads):
            try:
                proxy = ep.Proxy(
                    thread_idx=t,
                    gpu_buffer_addr=rdma_ptrs[gpu_id],
                    total_size=rdma_bytes,
                    rank=gpu_id,
                    node_idx=0,
                    local_rank=gpu_id,
                    num_experts=num_rdma_gpus,
                    num_ranks=num_virtual_ranks,
                    num_nodes=2,
                    use_normal_mode=True,
                    is_intranode=False,  # KEY: enable RDMA path
                    gpu_buffer_is_host_allocated=rdma_is_host[gpu_id],
                )
                proxies.append(proxy)
            except Exception as e:
                print(f"  GPU {gpu_id} thread {t}: FAILED: {e}")
                traceback.print_exc()
                return
        all_proxies.append(proxies)
        print(f"  GPU {gpu_id}: {len(proxies)} internode proxies created OK")
        # Check listen port (RDMA proxies should have TCP listen ports)
        for t, p in enumerate(proxies):
            port = p.get_listen_port()
            print(f"    thread {t}: listen_port={port}")

    print("[RDMA-INIT] Step 2 PASSED!")

    # Step 3: Register proxies and create Buffer with RDMA
    print("\n[RDMA-INIT] Step 3: Register proxies + create Buffer(rdma_bytes>0)...")
    for gpu_id in range(num_rdma_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        ep.register_proxies(gpu_id, all_proxies[gpu_id])

    buffers = []
    for gpu_id in range(num_rdma_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        try:
            buf = ep.Buffer(
                rank=gpu_id,
                num_ranks=num_virtual_ranks,
                num_nvl_bytes=nvl_bytes,
                num_rdma_bytes=rdma_bytes,
                low_latency_mode=False,
                explicitly_destroy=True,
                num_local_ranks=num_rdma_gpus,
            )
            buf.set_rdma_buffer(rdma_ptrs[gpu_id], rdma_is_host[gpu_id])
            buffers.append(buf)
            print(f"  GPU {gpu_id}: Buffer created with RDMA OK")
        except Exception as e:
            print(f"  GPU {gpu_id}: FAILED: {e}")
            traceback.print_exc()
            return
    print("[RDMA-INIT] Step 3 PASSED!")

    # Step 4: Sync buffers (intranode part)
    # num_ranks=num_virtual_ranks, so we must provide that many entries.
    # Pad with device 0's values for the virtual "remote" ranks.
    print("\n[RDMA-INIT] Step 4: sync_same_process (NVL part)...")
    device_ids = [buffers[i].get_local_device_id() for i in range(num_rdma_gpus)]
    buffer_ptrs = [buffers[i].get_local_buffer_ptr(0, False) for i in range(num_rdma_gpus)]
    for _ in range(num_virtual_ranks - num_rdma_gpus):
        device_ids.append(device_ids[0])
        buffer_ptrs.append(buffer_ptrs[0])
    print(f"  device_ids ({len(device_ids)}): {device_ids}")
    for gpu_id in range(num_rdma_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].sync_same_process(device_ids, buffer_ptrs)
        assert buffers[gpu_id].is_available(), f"Buffer {gpu_id} not available after sync!"
    print("[RDMA-INIT] Step 4 PASSED!")

    # Cleanup
    print("\n[RDMA-INIT] Cleaning up...")
    for gpu_id in range(num_rdma_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].destroy()
    for gpu_id in range(num_rdma_gpus):
        hip_set_device(gpu_id)
        for proxy in all_proxies[gpu_id]:
            try: proxy.stop()
            except: pass
        ep.unregister_proxy(gpu_id)
    cleanup_shm()

    print("\n" + "=" * 60)
    print("[RDMA-INIT] ALL TESTS PASSED!")
    print("  - allocate_rdma_buffer: OK per-GPU")
    print("  - Proxy(is_intranode=False): OK, ibv init succeeded")
    print("  - Buffer(rdma_bytes>0): OK")
    print("  - sync_same_process: OK")
    print("=" * 60)


if __name__ == "__main__":
    test_rdma_proxy_init()
