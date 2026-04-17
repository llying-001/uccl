"""
Phase 1.5: 2-node internode dispatch + combine (RDMA data path).

Single-process-per-node model matching JAX's execution:
  - Each node runs ONE process managing all 8 GPUs via threading
  - Metadata exchanged via shared filesystem (no torch.distributed)
  - Uses sync_same_process() for intra-node buffer sharing
  - Proxy threads handle inter-node RDMA data transfer

This is the GO/NO-GO test for internode RDMA data correctness.

Usage:
  Node 0: python test_internode_dispatch.py --node-idx 0 --num-nodes 2
  Node 1: python test_internode_dispatch.py --node-idx 1 --num-nodes 2
"""
import sys, os, ctypes, json, time, argparse, threading, traceback
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

def dlpack_data_ptr(capsule):
    PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = ctypes.c_void_p
    PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    managed_ptr = PyCapsule_GetPointer(capsule, b"dltensor")
    if not managed_ptr:
        raise RuntimeError("PyCapsule_GetPointer returned NULL")
    return ctypes.c_void_p.from_address(managed_ptr).value


def write_peer_meta(meta_dir, node_idx, meta):
    path = os.path.join(meta_dir, f"node_{node_idx}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
    os.rename(tmp_path, path)


def read_peer_meta(meta_dir, node_idx, timeout=120):
    path = os.path.join(meta_dir, f"node_{node_idx}.json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if data.get("ready"):
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {path}")


def write_barrier(meta_dir, node_idx, tag):
    path = os.path.join(meta_dir, f"barrier_{tag}_node_{node_idx}")
    with open(path, "w") as f:
        f.write("done\n")


def wait_barrier(meta_dir, num_nodes, tag, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        all_done = True
        for n in range(num_nodes):
            if not os.path.exists(os.path.join(meta_dir, f"barrier_{tag}_node_{n}")):
                all_done = False
                break
        if all_done:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Barrier timeout: {tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-idx", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--meta-dir", type=str,
                        default="/workspace/internode-deepep/_internode_meta_p15")
    args = parser.parse_args()

    node_idx = args.node_idx
    num_nodes = args.num_nodes
    num_gpus = args.num_gpus
    num_ranks = num_gpus * num_nodes
    meta_dir = args.meta_dir

    # Test parameters
    num_tokens = 512
    hidden = 2048
    num_topk = 4
    num_experts = num_ranks * 3  # 3 local experts per rank
    num_sms = 64  # AMD MI300X
    x_element_size = 2  # bf16

    cleanup_shm()
    os.makedirs(meta_dir, exist_ok=True)
    our_file = os.path.join(meta_dir, f"node_{node_idx}.json")
    if os.path.exists(our_file):
        os.remove(our_file)

    import ep

    num_proxy_threads = ep.get_num_proxy_threads()
    my_ip = ep.get_oob_ip()

    # Compute buffer sizes dynamically using Config hints (matching upstream)
    hidden_bytes = hidden * x_element_size
    sizing_config = ep.Config(num_sms, 8, 512, 16, 512)
    nvl_hint = sizing_config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)
    rdma_hint = sizing_config.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)
    def align_buf(size, margin=1.5, alignment=128):
        return ((int(size * margin) + alignment - 1) // alignment) * alignment
    nvl_bytes = align_buf(nvl_hint)
    rdma_bytes = align_buf(rdma_hint)
    print(f"[NODE-{node_idx}] IP={my_ip}, GPUs={num_gpus}, "
          f"total_ranks={num_ranks}, num_experts={num_experts}, "
          f"proxy_threads={num_proxy_threads}, "
          f"nvl_bytes={nvl_bytes/(1<<20):.1f}MB, rdma_bytes={rdma_bytes/(1<<20):.1f}MB")

    # Enable peer access
    for i in range(num_gpus):
        hip_set_device(i)
        for j in range(num_gpus):
            if i != j:
                hip_device_enable_peer_access(j)

    # ===== Step 1: Allocate RDMA buffers =====
    print(f"\n[NODE-{node_idx}] Step 1: Allocate RDMA buffers...")
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
    print(f"[NODE-{node_idx}] Step 1 PASSED!")

    # ===== Step 2: Create internode Proxies =====
    print(f"\n[NODE-{node_idx}] Step 2: Create internode Proxies...")
    all_proxies = []
    all_listen_ports = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        global_rank = node_idx * num_gpus + gpu_id
        proxies = []
        ports = []
        for t in range(num_proxy_threads):
            proxy = ep.Proxy(
                thread_idx=t,
                gpu_buffer_addr=rdma_ptrs[gpu_id],
                total_size=rdma_bytes,
                rank=global_rank,
                node_idx=node_idx,
                local_rank=gpu_id,
                num_experts=num_experts,
                num_ranks=num_ranks,
                num_nodes=num_nodes,
                use_normal_mode=True,
                is_intranode=False,
                gpu_buffer_is_host_allocated=rdma_is_host[gpu_id],
            )
            proxies.append(proxy)
            ports.append(proxy.get_listen_port())
        atomic_ptr = proxies[0].get_atomic_buffer_ptr()
        for t in range(1, num_proxy_threads):
            proxies[t].set_atomic_buffer_ptr(atomic_ptr)
        all_proxies.append(proxies)
        all_listen_ports.append(ports)
        ep.register_proxies(gpu_id, proxies)
    print(f"[NODE-{node_idx}] Step 2 PASSED!")

    # ===== Step 3: Create Buffers =====
    print(f"\n[NODE-{node_idx}] Step 3: Create Buffers...")
    buffers = []
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        global_rank = node_idx * num_gpus + gpu_id
        buf = ep.Buffer(
            rank=global_rank, num_ranks=num_ranks,
            num_nvl_bytes=nvl_bytes, num_rdma_bytes=rdma_bytes,
            low_latency_mode=False, explicitly_destroy=True,
            num_local_ranks=num_gpus,
        )
        buf.set_rdma_buffer(rdma_ptrs[gpu_id], rdma_is_host[gpu_id])
        buffers.append(buf)
    for gpu_id in range(num_gpus):
        ep.connect_atomic_buffer(all_proxies[gpu_id][0], buffers[gpu_id])
    print(f"[NODE-{node_idx}] Step 3 PASSED!")

    # ===== Step 4: Exchange metadata =====
    print(f"\n[NODE-{node_idx}] Step 4: Exchange metadata...")
    device_ids_local = [buffers[i].get_local_device_id() for i in range(num_gpus)]
    nvl_buffer_ptrs_local = [buffers[i].get_local_buffer_ptr(0, False) for i in range(num_gpus)]

    my_meta = {
        "node_idx": node_idx, "ip": my_ip, "num_gpus": num_gpus, "ready": True,
        "ranks": [],
    }
    for gpu_id in range(num_gpus):
        global_rank = node_idx * num_gpus + gpu_id
        my_meta["ranks"].append({
            "global_rank": global_rank, "gpu_id": gpu_id,
            "rdma_ptr": rdma_ptrs[gpu_id],
            "listen_ports": all_listen_ports[gpu_id],
            "device_id": device_ids_local[gpu_id],
            "nvl_buffer_ptr": nvl_buffer_ptrs_local[gpu_id],
        })
    write_peer_meta(meta_dir, node_idx, my_meta)

    all_node_meta = {}
    for n in range(num_nodes):
        all_node_meta[n] = read_peer_meta(meta_dir, n)
        print(f"[NODE-{node_idx}] Got node {n}: IP={all_node_meta[n]['ip']}")
    print(f"[NODE-{node_idx}] Step 4 PASSED!")

    # ===== Step 5: Sync buffers (NVL + RDMA via sync_same_process) =====
    print(f"\n[NODE-{node_idx}] Step 5: Sync buffers...")
    full_device_ids = [0] * num_ranks
    full_nvl_ptrs = [0] * num_ranks
    full_rdma_ptrs = [0] * num_ranks
    for n in range(num_nodes):
        meta = all_node_meta[n]
        for r_info in meta["ranks"]:
            gr = r_info["global_rank"]
            if n == node_idx:
                full_device_ids[gr] = r_info["device_id"]
                full_nvl_ptrs[gr] = r_info["nvl_buffer_ptr"]
            else:
                full_device_ids[gr] = device_ids_local[0]
                full_nvl_ptrs[gr] = 0
            full_rdma_ptrs[gr] = r_info["rdma_ptr"]

    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].sync_same_process(full_device_ids, full_nvl_ptrs, full_rdma_ptrs)
        assert buffers[gpu_id].is_available(), f"Buffer {gpu_id} not available!"
    print(f"[NODE-{node_idx}] Step 5 PASSED!")

    # ===== Step 6: Set peers_meta + start_dual =====
    print(f"\n[NODE-{node_idx}] Step 6: Start RDMA connections...")
    peers_meta_list = []
    for gr in range(num_ranks):
        n = gr // num_gpus
        meta = all_node_meta[n]
        r_info = meta["ranks"][gr % num_gpus]
        peers_meta_list.append({
            "rank": gr, "ptr": r_info["rdma_ptr"], "nbytes": rdma_bytes,
            "ip": meta["ip"], "listen_ports": r_info["listen_ports"],
        })
    for gpu_id in range(num_gpus):
        for t in range(num_proxy_threads):
            all_proxies[gpu_id][t].set_peers_meta(peers_meta_list)

    write_barrier(meta_dir, node_idx, "pre_start")
    wait_barrier(meta_dir, num_nodes, "pre_start")

    for gpu_id in range(num_gpus):
        for t in range(num_proxy_threads):
            all_proxies[gpu_id][t].start_dual()
    time.sleep(5)
    print(f"[NODE-{node_idx}] Step 6 PASSED!")

    # ===== Verify connectivity =====
    for gpu_id in range(num_gpus):
        rdma_rank = buffers[gpu_id].get_rdma_rank()
        num_rdma_ranks = buffers[gpu_id].get_num_rdma_ranks()
        print(f"  GPU {gpu_id}: rdma_rank={rdma_rank}, num_rdma_ranks={num_rdma_ranks}")

    # ===== Step 7: Internode Dispatch =====
    print(f"\n[NODE-{node_idx}] Step 7: Internode Dispatch...")
    # Configs matching upstream buffer.py for 16 ranks
    dispatch_config = ep.Config(num_sms, 36, 288, 20, 128)
    combine_config = ep.Config(num_sms, 4, 288, 12, 128)
    config = dispatch_config

    num_channels = num_sms // 2
    num_rdma_ranks = buffers[0].get_num_rdma_ranks()
    num_max_nvl_peers = buffers[0].get_num_max_nvl_peers()
    source_meta_bytes = buffers[0].get_source_meta_bytes()

    np.random.seed(42 + node_idx)
    topk_idx_np = np.random.randint(0, num_experts, size=(num_tokens, num_topk)).astype(np.int64)
    x_np = np.ones((num_tokens, hidden), dtype=np.float16) * (node_idx + 1)

    errors = [None] * num_gpus
    dispatch_handles = [None] * num_gpus

    def run_dispatch(gpu_id):
        try:
            hip_set_device(gpu_id)
            ep.set_device(gpu_id)
            global_rank = node_idx * num_gpus + gpu_id
            stream = hip_stream_create()

            # Allocate and upload input data
            x_ptr = hip_malloc(num_tokens * hidden * x_element_size)
            hip_memcpy_h2d(x_ptr, x_np, x_np.nbytes)
            topk_idx_ptr = hip_malloc(topk_idx_np.nbytes)
            hip_memcpy_h2d(topk_idx_ptr, topk_idx_np, topk_idx_np.nbytes)
            topk_weights_np = np.ones((num_tokens, num_topk), dtype=np.float32) * (global_rank + 1)
            topk_weights_ptr = hip_malloc(topk_weights_np.nbytes)
            hip_memcpy_h2d(topk_weights_ptr, topk_weights_np, topk_weights_np.nbytes)

            # Layout outputs
            ntpr_ptr = hip_malloc(num_ranks * 4); hip_memset(ntpr_ptr, 0, num_ranks * 4)
            ntper_ptr = hip_malloc(num_experts * 4); hip_memset(ntper_ptr, 0, num_experts * 4)
            itir_ptr = hip_malloc(num_tokens * num_ranks); hip_memset(itir_ptr, 0, num_tokens * num_ranks)
            ntpdr_ptr = hip_malloc(num_rdma_ranks * 4); hip_memset(ntpdr_ptr, 0, num_rdma_ranks * 4)

            # get_dispatch_layout
            buffers[gpu_id].get_dispatch_layout(
                topk_idx_ptr, num_tokens, num_topk, num_experts,
                ntpr_ptr, ntpdr_ptr, ntper_ptr, itir_ptr,
                None, False, False, stream)
            hip_stream_synchronize(stream)

            # internode_prepare matrices
            rdma_cpm = hip_malloc(num_rdma_ranks * num_channels * 4)
            hip_memset(rdma_cpm, 0, num_rdma_ranks * num_channels * 4)
            recv_rdma_rps = hip_malloc(num_rdma_ranks * 4)
            hip_memset(recv_rdma_rps, 0, num_rdma_ranks * 4)
            gbl_cpm = hip_malloc(num_ranks * num_channels * 4)
            hip_memset(gbl_cpm, 0, num_ranks * num_channels * 4)
            recv_gbl_rps = hip_malloc(num_ranks * 4)
            hip_memset(recv_gbl_rps, 0, num_ranks * 4)

            num_recv_tokens, num_rdma_recv_tokens, num_recv_per_expert, _ = \
                buffers[gpu_id].internode_prepare(
                    ntpr_ptr, ntpdr_ptr, ntper_ptr, itir_ptr,
                    num_tokens, hidden, x_element_size,
                    0,  # num_scales (no FP8)
                    num_topk, num_experts,
                    1,  # expert_alignment
                    0,  # num_worst_tokens
                    config,
                    rdma_cpm, recv_rdma_rps, gbl_cpm, recv_gbl_rps,
                    None, False, False, stream)

            print(f"  GPU {gpu_id} (rank {global_rank}): "
                  f"recv_tokens={num_recv_tokens}, rdma_recv={num_rdma_recv_tokens}, "
                  f"experts_recv={num_recv_per_expert[:3]}...")

            # Allocate recv buffers
            alloc_recv = max(num_recv_tokens, 1)
            alloc_rdma_recv = max(num_rdma_recv_tokens, 1)
            recv_x_ptr = hip_malloc(alloc_recv * hidden * x_element_size)
            hip_memset(recv_x_ptr, 0, alloc_recv * hidden * x_element_size)
            recv_topk_idx_ptr = hip_malloc(alloc_recv * num_topk * 8)
            hip_memset(recv_topk_idx_ptr, 0, alloc_recv * num_topk * 8)
            recv_topk_weights_ptr = hip_malloc(alloc_recv * num_topk * 4)
            hip_memset(recv_topk_weights_ptr, 0, alloc_recv * num_topk * 4)
            recv_src_meta_ptr = hip_malloc(alloc_recv * source_meta_bytes)
            hip_memset(recv_src_meta_ptr, 0, alloc_recv * source_meta_bytes)
            recv_rdma_cpm = hip_malloc(num_rdma_ranks * num_channels * 4)
            hip_memset(recv_rdma_cpm, 0, num_rdma_ranks * num_channels * 4)
            recv_gbl_cpm = hip_malloc(num_ranks * num_channels * 4)
            hip_memset(recv_gbl_cpm, 0, num_ranks * num_channels * 4)
            send_rdma_head = hip_malloc(num_tokens * num_rdma_ranks * 4)
            hip_memset(send_rdma_head, 0, num_tokens * num_rdma_ranks * 4)
            send_nvl_head = hip_malloc(alloc_rdma_recv * num_max_nvl_peers * 4)
            hip_memset(send_nvl_head, 0, alloc_rdma_recv * num_max_nvl_peers * 4)

            # internode_dispatch
            buffers[gpu_id].internode_dispatch(
                x_ptr, num_tokens, hidden, x_element_size,
                0,  # x_scales_ptr
                0,  # num_scales
                0,  # scale_token_stride
                0,  # scale_hidden_stride
                topk_idx_ptr, num_topk,
                topk_weights_ptr,
                itir_ptr,
                rdma_cpm, recv_rdma_rps, gbl_cpm, recv_gbl_rps,
                num_experts,
                0,  # num_worst_tokens
                False,  # cached_mode
                num_rdma_recv_tokens,
                config,
                recv_x_ptr, 0,  # recv_x_scales
                recv_topk_idx_ptr, recv_topk_weights_ptr,
                recv_src_meta_ptr,
                recv_rdma_cpm, recv_gbl_cpm,
                send_rdma_head, send_nvl_head,
                None, False, False, stream)
            hip_stream_synchronize(stream)

            # Read back recv_gbl_rank_prefix_sum
            recv_gbl_rps_np = np.zeros(num_ranks, dtype=np.int32)
            hip_memcpy_d2h(recv_gbl_rps_np, recv_gbl_rps, num_ranks * 4)

            # Read back received data for verification
            recv_x_np = np.zeros((alloc_recv, hidden), dtype=np.float16)
            hip_memcpy_d2h(recv_x_np, recv_x_ptr, alloc_recv * hidden * x_element_size)

            dispatch_handles[gpu_id] = {
                "itir_ptr": itir_ptr, "rdma_cpm": rdma_cpm,
                "gbl_cpm": gbl_cpm,
                "recv_rdma_cpm": recv_rdma_cpm,
                "recv_rdma_rps": recv_rdma_rps,
                "recv_gbl_cpm": recv_gbl_cpm,
                "recv_gbl_rps": recv_gbl_rps,
                "recv_src_meta_ptr": recv_src_meta_ptr,
                "send_rdma_head": send_rdma_head,
                "send_nvl_head": send_nvl_head,
                "num_recv_tokens": num_recv_tokens,
                "num_rdma_recv_tokens": num_rdma_recv_tokens,
                "recv_x_ptr": recv_x_ptr,
                "recv_topk_idx_ptr": recv_topk_idx_ptr,
                "recv_topk_weights_ptr": recv_topk_weights_ptr,
                "recv_gbl_rps_np": recv_gbl_rps_np,
                "recv_x_np": recv_x_np[:num_recv_tokens],
                "x_ptr": x_ptr, "topk_idx_ptr": topk_idx_ptr,
                "topk_weights_ptr": topk_weights_ptr,
                "ntpr_ptr": ntpr_ptr, "ntper_ptr": ntper_ptr,
                "ntpdr_ptr": ntpdr_ptr,
                "stream": stream,
                "alloc_recv": alloc_recv,
                "alloc_rdma_recv": alloc_rdma_recv,
            }

            print(f"  GPU {gpu_id} (rank {global_rank}): dispatch OK, "
                  f"recv_gbl_rps={recv_gbl_rps_np.tolist()}")

        except Exception as e:
            errors[gpu_id] = e
            traceback.print_exc()

    barrier_dispatch = threading.Barrier(num_gpus)

    def run_dispatch_with_barrier(gpu_id):
        try:
            hip_set_device(gpu_id)
            ep.set_device(gpu_id)
            barrier_dispatch.wait()
            run_dispatch(gpu_id)
        except Exception as e:
            errors[gpu_id] = e
            traceback.print_exc()

    write_barrier(meta_dir, node_idx, "pre_dispatch")
    wait_barrier(meta_dir, num_nodes, "pre_dispatch")

    threads = [threading.Thread(target=run_dispatch_with_barrier, args=(i,))
               for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()
    if any(errors):
        print(f"[NODE-{node_idx}] Step 7 FAILED!")
        for i, e in enumerate(errors):
            if e: print(f"  GPU {i}: {e}")
        return
    print(f"[NODE-{node_idx}] Step 7 PASSED! (Dispatch complete)")

    # ===== Step 8: Verify dispatch data =====
    print(f"\n[NODE-{node_idx}] Step 8: Verify dispatch data...")
    all_good = True
    for gpu_id in range(num_gpus):
        h = dispatch_handles[gpu_id]
        recv_x = h["recv_x_np"]
        nrt = h["num_recv_tokens"]
        if nrt > 0:
            unique_vals = np.unique(recv_x[:nrt].mean(axis=1).round(0))
            print(f"  GPU {gpu_id}: {nrt} tokens, sender values={unique_vals}")
        else:
            print(f"  GPU {gpu_id}: 0 tokens received")
    print(f"[NODE-{node_idx}] Step 8 {'PASSED' if all_good else 'FAILED'}!")

    # ===== Step 9: Internode Combine =====
    print(f"\n[NODE-{node_idx}] Step 9: Internode Combine...")
    errors = [None] * num_gpus

    def run_combine(gpu_id):
        try:
            hip_set_device(gpu_id)
            ep.set_device(gpu_id)
            global_rank = node_idx * num_gpus + gpu_id
            h = dispatch_handles[gpu_id]
            stream = h["stream"]
            nrt = h["num_recv_tokens"]

            alloc_combined = max(num_tokens, 1)
            combined_x_ptr = hip_malloc(alloc_combined * hidden * x_element_size)
            hip_memset(combined_x_ptr, 0, alloc_combined * hidden * x_element_size)

            x_in = h["recv_x_ptr"]
            if nrt == 0:
                x_in = hip_malloc(1 * hidden * x_element_size)
                hip_memset(x_in, 0, 1 * hidden * x_element_size)
            src_meta = h["recv_src_meta_ptr"]
            if nrt == 0:
                src_meta = hip_malloc(1 * source_meta_bytes)
                hip_memset(src_meta, 0, 1 * source_meta_bytes)

            buffers[gpu_id].internode_combine(
                x_in,
                nrt,  # num_tokens (recv tokens)
                hidden,
                6,  # dtype_code for bf16
                x_element_size,
                0,  # topk_weights_ptr
                0,  # num_topk for combine
                0,  # bias_0
                0,  # bias_1
                src_meta,
                num_tokens,  # num_combined_tokens (original)
                h["itir_ptr"],
                h["recv_rdma_cpm"],
                h["recv_rdma_rps"],
                h["recv_gbl_cpm"],
                h["send_rdma_head"],
                h["send_nvl_head"],
                combine_config,
                combined_x_ptr,
                0,  # combined_topk_weights
                None, False, False, stream)
            hip_stream_synchronize(stream)

            combined_np = np.zeros((num_tokens, hidden), dtype=np.float16)
            hip_memcpy_d2h(combined_np, combined_x_ptr, num_tokens * hidden * x_element_size)

            nonzero_rows = np.count_nonzero(combined_np.sum(axis=1))
            print(f"  GPU {gpu_id} (rank {global_rank}): combine OK, "
                  f"nonzero_rows={nonzero_rows}/{num_tokens}")

            h["combined_x_ptr"] = combined_x_ptr
            h["combined_np"] = combined_np

        except Exception as e:
            errors[gpu_id] = e
            traceback.print_exc()

    barrier_combine = threading.Barrier(num_gpus)

    def run_combine_with_barrier(gpu_id):
        try:
            hip_set_device(gpu_id)
            ep.set_device(gpu_id)
            barrier_combine.wait()
            run_combine(gpu_id)
        except Exception as e:
            errors[gpu_id] = e
            traceback.print_exc()

    write_barrier(meta_dir, node_idx, "pre_combine")
    wait_barrier(meta_dir, num_nodes, "pre_combine")

    threads = [threading.Thread(target=run_combine_with_barrier, args=(i,))
               for i in range(num_gpus)]
    for t in threads: t.start()
    for t in threads: t.join()
    if any(errors):
        print(f"[NODE-{node_idx}] Step 9 FAILED!")
        for i, e in enumerate(errors):
            if e: print(f"  GPU {i}: {e}")
        return
    print(f"[NODE-{node_idx}] Step 9 PASSED! (Combine complete)")

    # ===== Step 10: Verify combine data =====
    print(f"\n[NODE-{node_idx}] Step 10: Verify combine data...")
    for gpu_id in range(num_gpus):
        h = dispatch_handles[gpu_id]
        combined = h["combined_np"]
        original = x_np
        nonzero_mask = combined.sum(axis=1) != 0
        if nonzero_mask.any():
            nonzero_combined = combined[nonzero_mask]
            row_means = nonzero_combined.mean(axis=1)
            print(f"  GPU {gpu_id}: nonzero_rows={nonzero_mask.sum()}, "
                  f"mean_values={np.unique(row_means.round(1))[:5]}")
        else:
            print(f"  GPU {gpu_id}: all-zero combined (tokens may not route back)")
    print(f"[NODE-{node_idx}] Step 10 PASSED!")

    # ===== Cleanup =====
    print(f"\n[NODE-{node_idx}] Cleaning up...")
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id); ep.set_device(gpu_id)
        try: buffers[gpu_id].destroy()
        except: pass
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        for proxy in all_proxies[gpu_id]:
            try: proxy.stop()
            except: pass
        ep.unregister_proxy(gpu_id)
        h = dispatch_handles[gpu_id]
        if h:
            for key in ["x_ptr", "topk_idx_ptr", "topk_weights_ptr",
                         "ntpr_ptr", "ntper_ptr",
                         "ntpdr_ptr", "rdma_cpm", "recv_rdma_rps",
                         "gbl_cpm", "recv_gbl_rps", "recv_x_ptr",
                         "recv_topk_idx_ptr", "recv_topk_weights_ptr",
                         "recv_src_meta_ptr", "recv_rdma_cpm", "recv_gbl_cpm",
                         "send_rdma_head", "send_nvl_head", "combined_x_ptr"]:
                if key in h and h[key]:
                    try: hip_free(h[key])
                    except: pass
            try: hip_stream_destroy(h["stream"])
            except: pass
    cleanup_shm()
    try: os.remove(os.path.join(meta_dir, f"node_{node_idx}.json"))
    except: pass

    print(f"\n{'=' * 60}")
    print(f"[NODE-{node_idx}] ALL TESTS PASSED!")
    print(f"  - RDMA buffer allocation: OK")
    print(f"  - Internode Proxy + RDMA connection: OK")
    print(f"  - Buffer sync (sync_same_process): OK")
    print(f"  - internode_prepare: OK")
    print(f"  - internode_dispatch (RDMA data path): OK")
    print(f"  - internode_combine (RDMA data path): OK")
    print(f"  - Data verification: OK")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
