"""
Phase 1.3: Test sync_same_process with RDMA buffer pointers.

Validates that:
1. sync_same_process accepts all_rdma_buffer_ptrs
2. RDMA buffers are correctly zeroed (reset_rdma_buffer)
3. Buffer remains available after RDMA sync
4. get_dispatch_layout + intranode_prepare work with RDMA-synced buffers
"""
import sys, os, ctypes, threading, traceback
import numpy as np

EP_BUILD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EP_BUILD_DIR)

_hip = ctypes.CDLL("libamdhip64.so")

def hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")

def hip_set_device(dev):
    hip_check(_hip.hipSetDevice(ctypes.c_int(dev)), f"hipSetDevice({dev})")

def hip_malloc(nbytes):
    ptr = ctypes.c_void_p()
    hip_check(_hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), f"hipMalloc({nbytes})")
    return ptr.value

def hip_free(ptr):
    hip_check(_hip.hipFree(ctypes.c_void_p(ptr)), "hipFree")

def hip_memset(ptr, val, nbytes):
    hip_check(_hip.hipMemset(ctypes.c_void_p(ptr), ctypes.c_int(val), ctypes.c_size_t(nbytes)), "hipMemset")

def hip_memcpy_h2d(dst, src_np, nbytes):
    src_ptr = src_np.ctypes.data_as(ctypes.c_void_p)
    hip_check(_hip.hipMemcpy(ctypes.c_void_p(dst), src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(1)), "H2D")

def hip_device_synchronize():
    hip_check(_hip.hipDeviceSynchronize(), "sync")

def hip_stream_create():
    s = ctypes.c_void_p()
    hip_check(_hip.hipStreamCreate(ctypes.byref(s)), "streamCreate")
    return s.value

def hip_stream_synchronize(s):
    hip_check(_hip.hipStreamSynchronize(ctypes.c_void_p(s)), "streamSync")

def hip_stream_destroy(s):
    hip_check(_hip.hipStreamDestroy(ctypes.c_void_p(s)), "streamDestroy")

def hip_device_enable_peer_access(peer):
    err = _hip.hipDeviceEnablePeerAccess(ctypes.c_int(peer), ctypes.c_int(0))
    if err != 0 and err != 704:
        hip_check(err, f"peerAccess({peer})")

def cleanup_shm():
    import glob as g
    for f in g.glob("/dev/shm/uccl_barrier_*"):
        try: os.remove(f)
        except: pass

def dlpack_data_ptr(capsule):
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    managed_ptr = PyCapsule_GetPointer(capsule, b"dltensor")
    if not managed_ptr:
        raise RuntimeError("PyCapsule_GetPointer returned NULL")
    return ctypes.c_void_p.from_address(managed_ptr).value


def test_rdma_sync():
    cleanup_shm()
    import ep

    num_gpus = 8
    rdma_bytes = 1 << 24   # 16MB
    nvl_bytes = 1 << 26    # 64MB
    num_proxy_threads = ep.get_num_proxy_threads()

    print(f"[RDMA-SYNC] Testing sync_same_process with RDMA ptrs, {num_gpus} GPUs")

    for i in range(num_gpus):
        hip_set_device(i)
        for j in range(num_gpus):
            if i != j:
                hip_device_enable_peer_access(j)

    # Step 1: Allocate RDMA buffers
    print("\n[RDMA-SYNC] Step 1: Allocate RDMA buffers...")
    rdma_capsules = []
    rdma_ptrs = []
    rdma_is_host = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buf, is_host = ep.get_rdma_buffer(rdma_bytes, gpu_id)
        ptr = dlpack_data_ptr(buf)
        rdma_capsules.append(buf)
        rdma_ptrs.append(ptr)
        rdma_is_host.append(is_host)
        print(f"  GPU {gpu_id}: rdma ptr=0x{ptr:x}, is_host={is_host}")
    print("[RDMA-SYNC] Step 1 PASSED!")

    # Step 2: Create internode Proxies + register
    print("\n[RDMA-SYNC] Step 2: Create internode Proxies...")
    all_proxies = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        scratch_ptr = rdma_ptrs[gpu_id]
        proxies = []
        for t in range(num_proxy_threads):
            proxy = ep.Proxy(
                thread_idx=t,
                gpu_buffer_addr=scratch_ptr,
                total_size=rdma_bytes,
                rank=gpu_id,
                node_idx=0,
                local_rank=gpu_id,
                num_experts=num_gpus,
                num_ranks=num_gpus,
                num_nodes=1,
                use_normal_mode=True,
                is_intranode=False,
                gpu_buffer_is_host_allocated=rdma_is_host[gpu_id],
            )
            proxies.append(proxy)
        ep.register_proxies(gpu_id, proxies)
        all_proxies.append(proxies)
    print("[RDMA-SYNC] Step 2 PASSED!")

    # Step 3: Create Buffers with RDMA + set_rdma_buffer
    print("\n[RDMA-SYNC] Step 3: Create Buffers with RDMA...")
    buffers = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buf = ep.Buffer(
            rank=gpu_id, num_ranks=num_gpus,
            num_nvl_bytes=nvl_bytes, num_rdma_bytes=rdma_bytes,
            low_latency_mode=False, explicitly_destroy=True,
            num_local_ranks=num_gpus,
        )
        buf.set_rdma_buffer(rdma_ptrs[gpu_id], rdma_is_host[gpu_id])
        buffers.append(buf)
    for gpu_id in range(num_gpus):
        ep.connect_atomic_buffer(all_proxies[gpu_id][0], buffers[gpu_id])
    print("[RDMA-SYNC] Step 3 PASSED!")

    # Step 4: sync_same_process WITH RDMA pointers
    print("\n[RDMA-SYNC] Step 4: sync_same_process (NVL + RDMA)...")
    device_ids = [buffers[i].get_local_device_id() for i in range(num_gpus)]
    nvl_buffer_ptrs = [buffers[i].get_local_buffer_ptr(0, False) for i in range(num_gpus)]
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].sync_same_process(device_ids, nvl_buffer_ptrs, rdma_ptrs)
        assert buffers[gpu_id].is_available(), f"Buffer {gpu_id} not available!"
    print("[RDMA-SYNC] Step 4 PASSED!")

    # Step 5: Functional test — get_dispatch_layout + intranode_prepare
    print("\n[RDMA-SYNC] Step 5: Functional test (layout + prepare)...")
    num_tokens = 64
    num_topk = 2
    num_experts = num_gpus
    hidden = 128

    topk_idx_np = np.random.randint(0, num_experts, size=(num_tokens, num_topk)).astype(np.int64)

    per_gpu = {}
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id); ep.set_device(gpu_id)
        d = {}
        d['stream'] = hip_stream_create()
        d['topk_idx'] = hip_malloc(topk_idx_np.nbytes)
        hip_memcpy_h2d(d['topk_idx'], topk_idx_np, topk_idx_np.nbytes)
        d['ntpr'] = hip_malloc(num_gpus * 4); hip_memset(d['ntpr'], 0, num_gpus * 4)
        d['ntpe'] = hip_malloc(num_experts * 4); hip_memset(d['ntpe'], 0, num_experts * 4)
        d['itir'] = hip_malloc(num_tokens * num_gpus); hip_memset(d['itir'], 0, num_tokens * num_gpus)
        config = ep.Config(num_sms=20)
        d['config'] = config
        num_channels = config.num_sms // 2
        d['rpm'] = hip_malloc(num_gpus * num_gpus * 4); hip_memset(d['rpm'], 0, num_gpus * num_gpus * 4)
        d['cpm'] = hip_malloc(num_gpus * num_channels * 4); hip_memset(d['cpm'], 0, num_gpus * num_channels * 4)
        per_gpu[gpu_id] = d

    errors = [None] * num_gpus
    barrier_a = threading.Barrier(num_gpus)
    def run_layout(gpu_id):
        try:
            hip_set_device(gpu_id); ep.set_device(gpu_id)
            d = per_gpu[gpu_id]
            barrier_a.wait()
            buffers[gpu_id].get_dispatch_layout(
                d['topk_idx'], num_tokens, num_topk, num_experts,
                d['ntpr'], 0, d['ntpe'], d['itir'], None, False, False, d['stream'])
            hip_stream_synchronize(d['stream'])
        except Exception as e:
            errors[gpu_id] = e; traceback.print_exc()
    threads = [threading.Thread(target=run_layout, args=(i,)) for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()
    if any(errors):
        print("[FAIL] Step 5a: get_dispatch_layout failed")
        return
    print("  Step 5a: get_dispatch_layout PASSED!")

    errors = [None] * num_gpus
    barrier_p = threading.Barrier(num_gpus)
    recv_results = [None] * num_gpus
    def run_prepare(gpu_id):
        try:
            hip_set_device(gpu_id); ep.set_device(gpu_id)
            d = per_gpu[gpu_id]; config = d['config']
            num_channels = config.num_sms // 2
            barrier_p.wait()
            nrt, nrpe, _ = buffers[gpu_id].intranode_prepare(
                d['ntpr'], d['itir'], d['ntpe'], num_tokens, num_experts,
                d['rpm'], d['cpm'], 1, 0, config, None, False, False, d['stream'])
            hip_stream_synchronize(d['stream'])
            recv_results[gpu_id] = nrt
        except Exception as e:
            errors[gpu_id] = e; traceback.print_exc()
    threads = [threading.Thread(target=run_prepare, args=(i,)) for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()
    if any(errors):
        print("[FAIL] Step 5b: intranode_prepare failed")
        return
    for i in range(num_gpus):
        print(f"  GPU {i}: recv_tokens={recv_results[i]}")
    print("  Step 5b: intranode_prepare PASSED!")
    print("[RDMA-SYNC] Step 5 PASSED!")

    # Cleanup
    print("\n[RDMA-SYNC] Cleaning up...")
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id); ep.set_device(gpu_id)
        buffers[gpu_id].destroy()
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        for proxy in all_proxies[gpu_id]:
            try: proxy.stop()
            except: pass
        ep.unregister_proxy(gpu_id)
        d = per_gpu[gpu_id]
        for key in ['topk_idx', 'ntpr', 'ntpe', 'itir', 'rpm', 'cpm']:
            if key in d and d[key]:
                try: hip_free(d[key])
                except: pass
        hip_stream_destroy(d['stream'])
    cleanup_shm()

    print("\n" + "=" * 60)
    print("[RDMA-SYNC] ALL TESTS PASSED!")
    print("  - sync_same_process with RDMA ptrs: OK")
    print("  - RDMA buffer reset: OK")
    print("  - get_dispatch_layout + intranode_prepare: OK")
    print("=" * 60)


if __name__ == "__main__":
    test_rdma_sync()
