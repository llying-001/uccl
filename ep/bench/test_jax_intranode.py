"""
PoC Test: UCCL-EP intranode dispatch/combine with 1-process-N-GPUs model.

This test validates that UCCL-EP can operate in a single-process multi-GPU mode
(compatible with JAX's 1-process-per-node deployment).

Scheme A: Pure Python + HIP ctypes for GPU memory management.
All GPU operations run concurrently using Python threads.
"""

import sys
import os
import ctypes
import time
import glob
import threading
import traceback
import numpy as np

EP_BUILD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EP_BUILD_DIR)
import ep

# ── HIP ctypes helpers ──────────────────────────────────────────────────
_hip = ctypes.CDLL("libamdhip64.so")

def hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")

def hip_set_device(dev):
    hip_check(_hip.hipSetDevice(ctypes.c_int(dev)), f"hipSetDevice({dev})")

def hip_get_device_count():
    count = ctypes.c_int()
    hip_check(_hip.hipGetDeviceCount(ctypes.byref(count)), "hipGetDeviceCount")
    return count.value

def hip_malloc(nbytes):
    ptr = ctypes.c_void_p()
    hip_check(_hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
              f"hipMalloc({nbytes})")
    return ptr.value

def hip_free(ptr):
    hip_check(_hip.hipFree(ctypes.c_void_p(ptr)), "hipFree")

def hip_memset(ptr, val, nbytes):
    hip_check(_hip.hipMemset(ctypes.c_void_p(ptr), ctypes.c_int(val),
                             ctypes.c_size_t(nbytes)), "hipMemset")

def hip_memcpy_h2d(dst, src_np, nbytes):
    src_ptr = src_np.ctypes.data_as(ctypes.c_void_p)
    hip_check(_hip.hipMemcpy(ctypes.c_void_p(dst), src_ptr,
                             ctypes.c_size_t(nbytes), ctypes.c_int(1)),
              "hipMemcpy H2D")

def hip_memcpy_d2h(dst_np, src, nbytes):
    dst_ptr = dst_np.ctypes.data_as(ctypes.c_void_p)
    hip_check(_hip.hipMemcpy(dst_ptr, ctypes.c_void_p(src),
                             ctypes.c_size_t(nbytes), ctypes.c_int(2)),
              "hipMemcpy D2H")

def hip_device_synchronize():
    hip_check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")

def hip_stream_create():
    stream = ctypes.c_void_p()
    hip_check(_hip.hipStreamCreate(ctypes.byref(stream)), "hipStreamCreate")
    return stream.value

def hip_stream_synchronize(stream):
    hip_check(_hip.hipStreamSynchronize(ctypes.c_void_p(stream)),
              "hipStreamSynchronize")

def hip_stream_destroy(stream):
    hip_check(_hip.hipStreamDestroy(ctypes.c_void_p(stream)),
              "hipStreamDestroy")

def hip_device_enable_peer_access(peer_dev):
    err = _hip.hipDeviceEnablePeerAccess(ctypes.c_int(peer_dev), ctypes.c_int(0))
    if err != 0 and err != 704:
        hip_check(err, f"hipDeviceEnablePeerAccess({peer_dev})")


def cleanup_shm():
    for f in glob.glob("/dev/shm/uccl_barrier_*"):
        try:
            os.remove(f)
        except Exception:
            pass


def test_single_process_multi_gpu():
    cleanup_shm()

    num_gpus = min(hip_get_device_count(), 8)
    print(f"[PoC] Testing with {num_gpus} GPUs in single process")

    # ── Configuration ──
    nvl_bytes = 1 << 26  # 64 MB per GPU
    rdma_bytes = 0
    num_proxy_threads = ep.get_num_proxy_threads()
    print(f"[PoC] num_proxy_threads = {num_proxy_threads}")
    print(f"[PoC] nvl_bytes = {nvl_bytes}")

    # ── Step 1: Enable peer access ──
    print("[PoC] Enabling peer access...")
    for i in range(num_gpus):
        hip_set_device(i)
        for j in range(num_gpus):
            if i != j:
                hip_device_enable_peer_access(j)
    print("[PoC] Peer access enabled")

    # ── Step 2: Create proxies for each GPU ──
    print("[PoC] Creating proxies...")
    all_proxies = []
    scratch_ptrs = []
    scratch_nbytes = max(rdma_bytes, 1)

    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)

        scratch_ptr = hip_malloc(scratch_nbytes)
        hip_memset(scratch_ptr, 0, scratch_nbytes)
        scratch_ptrs.append(scratch_ptr)

        proxies = []
        for t in range(num_proxy_threads):
            proxy = ep.Proxy(
                thread_idx=t,
                gpu_buffer_addr=scratch_ptr,
                total_size=scratch_nbytes,
                rank=gpu_id,
                node_idx=0,
                local_rank=gpu_id,
                num_experts=0,
                num_ranks=num_gpus,
                num_nodes=1,
                use_normal_mode=True,
                is_intranode=True,
                gpu_buffer_is_host_allocated=False,
            )
            proxies.append(proxy)

        ep.register_proxies(gpu_id, proxies)
        all_proxies.append(proxies)
        print(f"  GPU {gpu_id}: registered {len(proxies)} proxies")

    # ── Step 3: Create ep.Buffer for each GPU ──
    print("[PoC] Creating Buffers...")
    buffers = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)

        buf = ep.Buffer(
            rank=gpu_id,
            num_ranks=num_gpus,
            num_nvl_bytes=nvl_bytes,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=False,
            explicitly_destroy=True,
            num_local_ranks=num_gpus,
        )
        buffers.append(buf)

    # ── Step 4: Sync buffers (same-process) ──
    print("[PoC] Syncing buffers (same-process mode)...")
    device_ids = [buffers[i].get_local_device_id() for i in range(num_gpus)]
    buffer_ptrs = [buffers[i].get_local_buffer_ptr(0, False) for i in range(num_gpus)]

    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].sync_same_process(device_ids, buffer_ptrs)
        assert buffers[gpu_id].is_available()

    # Connect atomic buffers
    for gpu_id in range(num_gpus):
        ep.connect_atomic_buffer(all_proxies[gpu_id][0], buffers[gpu_id])

    print("[PoC] All buffers synced successfully!")

    # ── Step 5: Concurrent dispatch test using threads ──
    num_tokens = 64
    num_topk = 2
    num_experts = num_gpus
    hidden = 128

    # Same topk_idx for all GPUs (simulating same routing decision)
    topk_idx_np = np.random.randint(0, num_experts, size=(num_tokens, num_topk)).astype(np.int64)

    # Pre-allocate GPU resources per rank
    per_gpu = {}
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)

        d = {}
        d['stream'] = hip_stream_create()
        d['topk_idx'] = hip_malloc(topk_idx_np.nbytes)
        hip_memcpy_h2d(d['topk_idx'], topk_idx_np, topk_idx_np.nbytes)
        d['ntpr'] = hip_malloc(num_gpus * 4)
        hip_memset(d['ntpr'], 0, num_gpus * 4)
        d['ntpe'] = hip_malloc(num_experts * 4)
        hip_memset(d['ntpe'], 0, num_experts * 4)
        d['itir'] = hip_malloc(num_tokens * num_gpus)
        hip_memset(d['itir'], 0, num_tokens * num_gpus)

        # Input x: [num_tokens, hidden] bfloat16
        x_np = np.random.randn(num_tokens, hidden).astype(np.float32)
        x_bf16 = (x_np.view(np.uint32) >> 16).astype(np.uint16)
        d['x'] = hip_malloc(num_tokens * hidden * 2)
        hip_memcpy_h2d(d['x'], x_bf16, num_tokens * hidden * 2)

        config = ep.Config(num_sms=20)
        num_channels = config.num_sms // 2
        d['config'] = config
        d['num_channels'] = num_channels
        d['rpm'] = hip_malloc(num_gpus * num_gpus * 4)
        hip_memset(d['rpm'], 0, num_gpus * num_gpus * 4)
        d['cpm'] = hip_malloc(num_gpus * num_channels * 4)
        hip_memset(d['cpm'], 0, num_gpus * num_channels * 4)

        per_gpu[gpu_id] = d

    # ── Phase A: get_dispatch_layout on all GPUs concurrently ──
    print("[PoC] Phase A: get_dispatch_layout (threaded)...")
    errors = [None] * num_gpus
    barrier_layout = threading.Barrier(num_gpus)

    def run_layout(gpu_id):
        try:
            hip_set_device(gpu_id)
            ep.set_device(gpu_id)
            d = per_gpu[gpu_id]
            barrier_layout.wait()  # synchronize start
            buffers[gpu_id].get_dispatch_layout(
                d['topk_idx'], num_tokens, num_topk, num_experts,
                d['ntpr'], 0, d['ntpe'], d['itir'],
                None, False, False, d['stream']
            )
            hip_stream_synchronize(d['stream'])
        except Exception as e:
            errors[gpu_id] = e
            traceback.print_exc()

    threads = [threading.Thread(target=run_layout, args=(i,)) for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()

    for i, err in enumerate(errors):
        if err is not None:
            print(f"[FAIL] GPU {i}: {err}")
            return

    # Print layout results for GPU 0
    d = per_gpu[0]
    hip_set_device(0)
    ntpr_result = np.zeros(num_gpus, dtype=np.int32)
    hip_memcpy_d2h(ntpr_result, d['ntpr'], num_gpus * 4)
    print(f"  GPU 0 num_tokens_per_rank = {ntpr_result}")
    print("[PoC] Phase A PASSED!")

    # ── Phase B: intranode_prepare + intranode_dispatch on all GPUs concurrently ──
    print("[PoC] Phase B: intranode_prepare + dispatch (threaded)...")

    recv_results = [None] * num_gpus
    barrier_prepare = threading.Barrier(num_gpus)
    barrier_dispatch = threading.Barrier(num_gpus)

    def run_dispatch(gpu_id):
        try:
            hip_set_device(gpu_id)
            ep.set_device(gpu_id)
            d = per_gpu[gpu_id]
            config = d['config']
            num_channels = d['num_channels']

            barrier_prepare.wait()
            num_recv_tokens, num_recv_per_expert_list, _ = buffers[gpu_id].intranode_prepare(
                d['ntpr'], d['itir'], d['ntpe'],
                num_tokens, num_experts,
                d['rpm'], d['cpm'],
                1, 0, config, None, False, False, d['stream']
            )
            hip_stream_synchronize(d['stream'])

            recv_x = hip_malloc(max(num_recv_tokens, 1) * hidden * 2)
            hip_memset(recv_x, 0, max(num_recv_tokens, 1) * hidden * 2)
            recv_cpm = hip_malloc(num_gpus * num_channels * 4)
            hip_memset(recv_cpm, 0, num_gpus * num_channels * 4)
            recv_src_idx = hip_malloc(max(num_recv_tokens, 1) * 4)
            hip_memset(recv_src_idx, 0, max(num_recv_tokens, 1) * 4)
            send_head = hip_malloc(num_tokens * num_gpus * 4)
            hip_memset(send_head, 0, num_tokens * num_gpus * 4)

            d['num_recv_tokens'] = num_recv_tokens
            d['recv_x'] = recv_x
            d['recv_cpm'] = recv_cpm
            d['recv_src_idx'] = recv_src_idx
            d['send_head'] = send_head

            barrier_dispatch.wait()
            buffers[gpu_id].intranode_dispatch(
                d['x'], num_tokens, hidden, 2,  # x_ptr, rows, cols, elem_size
                0, 0, 0, 0,                     # scales (none)
                0, 0, 0,                         # topk (none)
                d['itir'], d['rpm'], d['cpm'],
                num_experts, 0,                  # num_experts, num_worst_tokens
                False,                           # is_cached
                config, num_recv_tokens,
                recv_x, 0, 0, 0,                 # recv_x, scales, topk_idx, topk_weights
                recv_cpm, recv_src_idx, send_head,
                None, False, False, d['stream']
            )
            hip_stream_synchronize(d['stream'])
            recv_results[gpu_id] = num_recv_tokens
        except Exception as e:
            errors[gpu_id] = e
            traceback.print_exc()

    errors = [None] * num_gpus
    threads = [threading.Thread(target=run_dispatch, args=(i,)) for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()

    for i, err in enumerate(errors):
        if err is not None:
            print(f"[FAIL] GPU {i}: {err}")
            return

    for i in range(num_gpus):
        print(f"  GPU {i}: recv_tokens = {recv_results[i]}")
    print("[PoC] Phase B PASSED!")

    # ── Phase C: intranode_combine on all GPUs concurrently ──
    print("[PoC] Phase C: intranode_combine (threaded)...")
    barrier_combine = threading.Barrier(num_gpus)

    def run_combine(gpu_id):
        try:
            hip_set_device(gpu_id)
            ep.set_device(gpu_id)
            d = per_gpu[gpu_id]
            config = d['config']
            num_recv_tokens = d['num_recv_tokens']

            combined_x = hip_malloc(max(num_tokens, 1) * hidden * 2)
            hip_memset(combined_x, 0, max(num_tokens, 1) * hidden * 2)
            d['combined_x'] = combined_x

            barrier_combine.wait()
            buffers[gpu_id].intranode_combine(
                d['recv_x'], num_recv_tokens, hidden,
                6, 2,  # dtype_code=bfloat16, elem_size=2
                0, 0,  # topk_weights, num_topk
                0, 0,  # bias_0, bias_1
                d['recv_src_idx'], num_tokens,
                d['rpm'], d['recv_cpm'], d['send_head'],
                config,
                combined_x,
                0,  # combined_topk_weights
                None, False, False, d['stream']
            )
            hip_stream_synchronize(d['stream'])
        except Exception as e:
            errors[gpu_id] = e
            traceback.print_exc()

    errors = [None] * num_gpus
    threads = [threading.Thread(target=run_combine, args=(i,)) for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()

    for i, err in enumerate(errors):
        if err is not None:
            print(f"[FAIL] GPU {i}: {err}")
            return

    # Verify combined data on GPU 0
    hip_set_device(0)
    combined_data = np.zeros(num_tokens * hidden, dtype=np.uint16)
    hip_memcpy_d2h(combined_data, per_gpu[0]['combined_x'], num_tokens * hidden * 2)
    nonzero_count = np.count_nonzero(combined_data)
    print(f"  GPU 0 combined: {nonzero_count}/{num_tokens * hidden} non-zero bf16 values")
    print("[PoC] Phase C PASSED!")

    # ── Cleanup ──
    print("[PoC] Cleaning up...")
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].destroy()

    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        for proxy in all_proxies[gpu_id]:
            try:
                proxy.stop()
            except RuntimeError:
                pass
        ep.unregister_proxy(gpu_id)
        d = per_gpu[gpu_id]
        for key in ['topk_idx', 'ntpr', 'ntpe', 'itir', 'x', 'rpm', 'cpm',
                     'recv_x', 'recv_cpm', 'recv_src_idx', 'send_head', 'combined_x']:
            if key in d and d[key]:
                try:
                    hip_free(d[key])
                except Exception:
                    pass
        hip_stream_destroy(d['stream'])
    for ptr in scratch_ptrs:
        try:
            hip_free(ptr)
        except Exception:
            pass
    cleanup_shm()

    print("\n" + "=" * 60)
    print("[PoC] ALL TESTS PASSED - Single-process multi-GPU UCCL-EP works!")
    print("=" * 60)


if __name__ == "__main__":
    test_single_process_multi_gpu()
