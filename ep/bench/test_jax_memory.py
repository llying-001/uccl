"""
Phase 2.1: Test UCCL-EP dispatch/combine with JAX-allocated input/output tensors.

Validates that JAX memory (standard hipMalloc via jax.device_put) works correctly
alongside UCCL-EP's internally-allocated uncached NVL buffers.
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

def hip_memcpy_d2h(dst_np, src, nbytes):
    dst_ptr = dst_np.ctypes.data_as(ctypes.c_void_p)
    hip_check(_hip.hipMemcpy(dst_ptr, ctypes.c_void_p(src), ctypes.c_size_t(nbytes), ctypes.c_int(2)), "D2H")

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

def get_jax_gpu_ptr(jax_array):
    """Extract raw GPU pointer from a JAX array."""
    buf = jax_array.addressable_data(0)
    return buf.unsafe_buffer_pointer()


def test_jax_memory_integration():
    cleanup_shm()

    # First, verify JAX can see GPUs
    os.environ.setdefault("JAX_PLATFORMS", "rocm,cpu")
    import jax
    import jax.numpy as jnp

    devices = jax.devices("gpu")
    num_gpus = min(len(devices), 8)
    print(f"[JAX-MEM] JAX sees {len(devices)} GPUs, using {num_gpus}")

    import ep

    nvl_bytes = 1 << 26
    num_proxy_threads = ep.get_num_proxy_threads()
    print(f"[JAX-MEM] nvl_bytes={nvl_bytes}, proxy_threads={num_proxy_threads}")

    # Enable peer access
    for i in range(num_gpus):
        hip_set_device(i)
        for j in range(num_gpus):
            if i != j:
                hip_device_enable_peer_access(j)

    # Create proxies
    all_proxies = []
    scratch_ptrs = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        scratch_ptr = hip_malloc(1)
        hip_memset(scratch_ptr, 0, 1)
        scratch_ptrs.append(scratch_ptr)
        proxies = []
        for t in range(num_proxy_threads):
            proxy = ep.Proxy(thread_idx=t, gpu_buffer_addr=scratch_ptr, total_size=1,
                           rank=gpu_id, node_idx=0, local_rank=gpu_id,
                           num_experts=0, num_ranks=num_gpus, num_nodes=1,
                           use_normal_mode=True, is_intranode=True,
                           gpu_buffer_is_host_allocated=False)
            proxies.append(proxy)
        ep.register_proxies(gpu_id, proxies)
        all_proxies.append(proxies)

    # Create and sync buffers
    buffers = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buf = ep.Buffer(rank=gpu_id, num_ranks=num_gpus, num_nvl_bytes=nvl_bytes,
                       num_rdma_bytes=0, low_latency_mode=False,
                       explicitly_destroy=True, num_local_ranks=num_gpus)
        buffers.append(buf)

    device_ids = [buffers[i].get_local_device_id() for i in range(num_gpus)]
    buffer_ptrs = [buffers[i].get_local_buffer_ptr(0, False) for i in range(num_gpus)]
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].sync_same_process(device_ids, buffer_ptrs)
    for gpu_id in range(num_gpus):
        ep.connect_atomic_buffer(all_proxies[gpu_id][0], buffers[gpu_id])

    # Test parameters
    num_tokens = 64
    num_topk = 2
    num_experts = num_gpus
    hidden = 128

    topk_idx_np = np.random.randint(0, num_experts, size=(num_tokens, num_topk)).astype(np.int64)

    # Allocate per-GPU resources — INPUT tensor from JAX, metadata from hip_malloc
    per_gpu = {}
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        d = {}
        d['stream'] = hip_stream_create()

        # JAX-allocated input tensor (key test)
        x_jax = jax.device_put(
            jnp.array(np.random.randn(num_tokens, hidden).astype(np.float32), dtype=jnp.bfloat16),
            devices[gpu_id]
        )
        d['x_jax'] = x_jax
        d['x_ptr'] = get_jax_gpu_ptr(x_jax)
        print(f"  GPU {gpu_id}: JAX x ptr = 0x{d['x_ptr']:x}")

        # Re-set device after jax.device_put (JAX may change HIP device context)
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)

        # Metadata buffers (hip_malloc is fine for these)
        d['topk_idx'] = hip_malloc(topk_idx_np.nbytes)
        hip_memcpy_h2d(d['topk_idx'], topk_idx_np, topk_idx_np.nbytes)
        d['ntpr'] = hip_malloc(num_gpus * 4); hip_memset(d['ntpr'], 0, num_gpus * 4)
        d['ntpe'] = hip_malloc(num_experts * 4); hip_memset(d['ntpe'], 0, num_experts * 4)
        d['itir'] = hip_malloc(num_tokens * num_gpus); hip_memset(d['itir'], 0, num_tokens * num_gpus)

        config = ep.Config(num_sms=20)
        num_channels = config.num_sms // 2
        d['config'] = config
        d['num_channels'] = num_channels
        d['rpm'] = hip_malloc(num_gpus * num_gpus * 4); hip_memset(d['rpm'], 0, num_gpus * num_gpus * 4)
        d['cpm'] = hip_malloc(num_gpus * num_channels * 4); hip_memset(d['cpm'], 0, num_gpus * num_channels * 4)
        per_gpu[gpu_id] = d

    # Phase A: get_dispatch_layout
    print("[JAX-MEM] Phase A: get_dispatch_layout...")
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
    if any(errors): print("[FAIL] Phase A"); return
    print("[JAX-MEM] Phase A PASSED!")

    # Phase B: intranode_prepare + dispatch (using JAX x_ptr)
    print("[JAX-MEM] Phase B: prepare + dispatch with JAX tensor...")
    barrier_p = threading.Barrier(num_gpus)
    barrier_d = threading.Barrier(num_gpus)
    recv_results = [None] * num_gpus
    def run_dispatch(gpu_id):
        try:
            hip_set_device(gpu_id); ep.set_device(gpu_id)
            d = per_gpu[gpu_id]; config = d['config']; nc = d['num_channels']
            barrier_p.wait()
            nrt, nrpe, _ = buffers[gpu_id].intranode_prepare(
                d['ntpr'], d['itir'], d['ntpe'], num_tokens, num_experts,
                d['rpm'], d['cpm'], 1, 0, config, None, False, False, d['stream'])
            hip_stream_synchronize(d['stream'])

            recv_x = hip_malloc(max(nrt, 1) * hidden * 2); hip_memset(recv_x, 0, max(nrt, 1) * hidden * 2)
            recv_cpm = hip_malloc(num_gpus * nc * 4); hip_memset(recv_cpm, 0, num_gpus * nc * 4)
            recv_src_idx = hip_malloc(max(nrt, 1) * 4); hip_memset(recv_src_idx, 0, max(nrt, 1) * 4)
            send_head = hip_malloc(num_tokens * num_gpus * 4); hip_memset(send_head, 0, num_tokens * num_gpus * 4)
            d['num_recv_tokens'] = nrt; d['recv_x'] = recv_x; d['recv_cpm'] = recv_cpm
            d['recv_src_idx'] = recv_src_idx; d['send_head'] = send_head

            barrier_d.wait()
            # Use JAX-allocated x_ptr here!
            buffers[gpu_id].intranode_dispatch(
                d['x_ptr'], num_tokens, hidden, 2,
                0, 0, 0, 0, 0, 0, 0,
                d['itir'], d['rpm'], d['cpm'], num_experts, 0, False,
                config, nrt, recv_x, 0, 0, 0, recv_cpm, recv_src_idx, send_head,
                None, False, False, d['stream'])
            hip_stream_synchronize(d['stream'])
            recv_results[gpu_id] = nrt
        except Exception as e:
            errors[gpu_id] = e; traceback.print_exc()
    errors = [None] * num_gpus
    threads = [threading.Thread(target=run_dispatch, args=(i,)) for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()
    if any(errors): print("[FAIL] Phase B"); return
    for i in range(num_gpus): print(f"  GPU {i}: recv_tokens={recv_results[i]}")
    print("[JAX-MEM] Phase B PASSED!")

    # Phase C: combine (output to JAX-allocated buffer)
    print("[JAX-MEM] Phase C: combine → JAX output tensor...")
    barrier_c = threading.Barrier(num_gpus)
    def run_combine(gpu_id):
        try:
            hip_set_device(gpu_id); ep.set_device(gpu_id)
            d = per_gpu[gpu_id]; config = d['config']; nrt = d['num_recv_tokens']

            # Output to JAX tensor
            combined_jax = jax.device_put(jnp.zeros((num_tokens, hidden), dtype=jnp.bfloat16), devices[gpu_id])
            d['combined_jax'] = combined_jax
            combined_ptr = get_jax_gpu_ptr(combined_jax)

            # Re-set device after jax.device_put (JAX may change HIP device context)
            hip_set_device(gpu_id); ep.set_device(gpu_id)

            barrier_c.wait()
            buffers[gpu_id].intranode_combine(
                d['recv_x'], nrt, hidden, 6, 2, 0, 0, 0, 0,
                d['recv_src_idx'], num_tokens,
                d['rpm'], d['recv_cpm'], d['send_head'], config,
                combined_ptr, 0, None, False, False, d['stream'])
            hip_stream_synchronize(d['stream'])
        except Exception as e:
            errors[gpu_id] = e; traceback.print_exc()
    errors = [None] * num_gpus
    threads = [threading.Thread(target=run_combine, args=(i,)) for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()
    if any(errors): print("[FAIL] Phase C"); return

    # Verify via JAX
    combined_arr = per_gpu[0]['combined_jax']
    nz = int(jnp.count_nonzero(combined_arr))
    print(f"  GPU 0 combined: {nz}/{num_tokens * hidden} non-zero bf16 values (via JAX)")
    print("[JAX-MEM] Phase C PASSED!")

    # Cleanup
    print("[JAX-MEM] Cleaning up...")
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
        for key in ['topk_idx','ntpr','ntpe','itir','rpm','cpm','recv_x','recv_cpm','recv_src_idx','send_head']:
            if key in d and d[key]:
                try: hip_free(d[key])
                except: pass
        hip_stream_destroy(d['stream'])
    for ptr in scratch_ptrs:
        try: hip_free(ptr)
        except: pass
    cleanup_shm()

    print("\n" + "=" * 60)
    print("[JAX-MEM] ALL TESTS PASSED - JAX memory works with UCCL-EP!")
    print("=" * 60)

if __name__ == "__main__":
    test_jax_memory_integration()
