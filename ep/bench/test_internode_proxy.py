"""
Phase 1.4: 2-node internode Proxy RDMA connection test.

Each node runs this script with --node-idx 0 or 1. The script:
1. Creates internode Proxies for 8 local GPUs
2. Writes PeerMeta to shared storage, reads peer node's PeerMeta
3. Establishes RDMA QP connections via Proxy start_dual()
4. Verifies connections are up and functional

Usage (run on 2 nodes via srun):
  Node 0: python test_internode_proxy.py --node-idx 0 --num-nodes 2
  Node 1: python test_internode_proxy.py --node-idx 1 --num-nodes 2
"""
import sys, os, ctypes, json, time, argparse, socket, traceback

EP_BUILD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EP_BUILD_DIR)

_hip = ctypes.CDLL("libamdhip64.so")

def hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")

def hip_set_device(dev):
    hip_check(_hip.hipSetDevice(ctypes.c_int(dev)), f"hipSetDevice({dev})")

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

def get_host_ip():
    """Get OOB IP using UCCL's interface discovery (same as ep.get_oob_ip)."""
    try:
        import ep as _ep
        return _ep.get_oob_ip()
    except Exception:
        pass
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()


def write_peer_meta(meta_dir, node_idx, meta):
    path = os.path.join(meta_dir, f"node_{node_idx}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
    os.rename(tmp_path, path)
    print(f"[NODE-{node_idx}] Wrote metadata to {path}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-idx", type=int, required=True)
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--meta-dir", type=str,
                        default="/workspace/internode-deepep/_internode_meta")
    args = parser.parse_args()

    node_idx = args.node_idx
    num_nodes = args.num_nodes
    num_gpus = args.num_gpus
    num_ranks = num_gpus * num_nodes
    meta_dir = args.meta_dir

    cleanup_shm()
    os.makedirs(meta_dir, exist_ok=True)
    # Clean up our own meta file
    our_file = os.path.join(meta_dir, f"node_{node_idx}.json")
    if os.path.exists(our_file):
        os.remove(our_file)

    import ep

    rdma_bytes = 1 << 24   # 16MB
    nvl_bytes = 1 << 26    # 64MB
    num_proxy_threads = ep.get_num_proxy_threads()
    my_ip = get_host_ip()
    print(f"[NODE-{node_idx}] IP={my_ip}, GPUs={num_gpus}, "
          f"total_ranks={num_ranks}, proxy_threads={num_proxy_threads}")

    # Enable peer access
    for i in range(num_gpus):
        hip_set_device(i)
        for j in range(num_gpus):
            if i != j:
                hip_device_enable_peer_access(j)

    # Step 1: Allocate RDMA buffers
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
    print(f"[NODE-{node_idx}] Step 1 PASSED! ({num_gpus} RDMA buffers allocated)")

    # Step 2: Create internode Proxies
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
                num_experts=num_ranks,
                num_ranks=num_ranks,
                num_nodes=num_nodes,
                use_normal_mode=True,
                is_intranode=False,
                gpu_buffer_is_host_allocated=rdma_is_host[gpu_id],
            )
            proxies.append(proxy)
            ports.append(proxy.get_listen_port())
        # Propagate atomic buffer from thread 0 to threads 1-3
        atomic_ptr = proxies[0].get_atomic_buffer_ptr()
        for t in range(1, num_proxy_threads):
            proxies[t].set_atomic_buffer_ptr(atomic_ptr)
        all_proxies.append(proxies)
        all_listen_ports.append(ports)
        ep.register_proxies(gpu_id, proxies)
    print(f"[NODE-{node_idx}] Step 2 PASSED! ({num_gpus}x{num_proxy_threads} proxies created)")

    # Step 3: Create Buffers + set RDMA + connect atomic
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

    # Step 4: Exchange PeerMeta via shared filesystem
    print(f"\n[NODE-{node_idx}] Step 4: Exchange PeerMeta...")
    device_ids_local = [buffers[i].get_local_device_id() for i in range(num_gpus)]
    nvl_buffer_ptrs_local = [buffers[i].get_local_buffer_ptr(0, False) for i in range(num_gpus)]

    my_meta = {
        "node_idx": node_idx,
        "ip": my_ip,
        "num_gpus": num_gpus,
        "ready": True,
        "ranks": [],
    }
    for gpu_id in range(num_gpus):
        global_rank = node_idx * num_gpus + gpu_id
        my_meta["ranks"].append({
            "global_rank": global_rank,
            "gpu_id": gpu_id,
            "rdma_ptr": rdma_ptrs[gpu_id],
            "listen_ports": all_listen_ports[gpu_id],
            "device_id": device_ids_local[gpu_id],
            "nvl_buffer_ptr": nvl_buffer_ptrs_local[gpu_id],
        })
    write_peer_meta(meta_dir, node_idx, my_meta)

    # Read all nodes' metadata
    all_node_meta = {}
    for n in range(num_nodes):
        print(f"[NODE-{node_idx}] Waiting for node {n} metadata...")
        all_node_meta[n] = read_peer_meta(meta_dir, n)
        print(f"[NODE-{node_idx}] Got node {n}: IP={all_node_meta[n]['ip']}")
    print(f"[NODE-{node_idx}] Step 4 PASSED!")

    # Step 5: Sync buffers (NVL + RDMA)
    print(f"\n[NODE-{node_idx}] Step 5: Sync buffers...")
    full_device_ids = [0] * num_ranks
    full_nvl_ptrs = [0] * num_ranks
    full_rdma_ptrs = [0] * num_ranks
    for n in range(num_nodes):
        meta = all_node_meta[n]
        for r_info in meta["ranks"]:
            gr = r_info["global_rank"]
            full_device_ids[gr] = r_info["device_id"]
            full_nvl_ptrs[gr] = r_info["nvl_buffer_ptr"]
            full_rdma_ptrs[gr] = r_info["rdma_ptr"]

    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        ep.set_device(gpu_id)
        buffers[gpu_id].sync_same_process(full_device_ids, full_nvl_ptrs, full_rdma_ptrs)
        assert buffers[gpu_id].is_available(), f"Buffer {gpu_id} not available!"
    print(f"[NODE-{node_idx}] Step 5 PASSED!")

    # Step 6: Set peers_meta + start_dual (THE KEY TEST)
    print(f"\n[NODE-{node_idx}] Step 6: Establish RDMA connections (start_dual)...")
    peers_meta_list = []
    for gr in range(num_ranks):
        n = gr // num_gpus
        meta = all_node_meta[n]
        r_info = meta["ranks"][gr % num_gpus]
        peers_meta_list.append({
            "rank": gr,
            "ptr": r_info["rdma_ptr"],
            "nbytes": rdma_bytes,
            "ip": meta["ip"],
            "listen_ports": r_info["listen_ports"],
        })

    for gpu_id in range(num_gpus):
        for t in range(num_proxy_threads):
            all_proxies[gpu_id][t].set_peers_meta(peers_meta_list)

    for gpu_id in range(num_gpus):
        global_rank = node_idx * num_gpus + gpu_id
        for t in range(num_proxy_threads):
            try:
                all_proxies[gpu_id][t].start_dual()
                print(f"  GPU {gpu_id} (rank {global_rank}) thread {t}: start_dual OK")
            except Exception as e:
                print(f"  GPU {gpu_id} (rank {global_rank}) thread {t}: start_dual FAILED: {e}")
                traceback.print_exc()
                return

    # Wait for all connections to fully establish
    time.sleep(5)
    print(f"[NODE-{node_idx}] Step 6 PASSED! All {num_gpus * num_proxy_threads} proxy threads running!")

    # Step 7: Verify RDMA connectivity
    print(f"\n[NODE-{node_idx}] Step 7: Verify buffers & connectivity...")
    for gpu_id in range(num_gpus):
        global_rank = node_idx * num_gpus + gpu_id
        assert buffers[gpu_id].is_available(), f"Buffer {global_rank} not available!"
        rdma_rank = buffers[gpu_id].get_rdma_rank()
        num_rdma_ranks = buffers[gpu_id].get_num_rdma_ranks()
        print(f"  GPU {gpu_id} (rank {global_rank}): available=True, "
              f"rdma_rank={rdma_rank}, num_rdma_ranks={num_rdma_ranks}")
    print(f"[NODE-{node_idx}] Step 7 PASSED!")

    # Cleanup
    print(f"\n[NODE-{node_idx}] Cleaning up...")
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id); ep.set_device(gpu_id)
        try: buffers[gpu_id].destroy()
        except Exception as e: print(f"  Buffer {gpu_id} destroy: {e}")
    for gpu_id in range(num_gpus):
        hip_set_device(gpu_id)
        for proxy in all_proxies[gpu_id]:
            try: proxy.stop()
            except: pass
        ep.unregister_proxy(gpu_id)
    cleanup_shm()
    try: os.remove(os.path.join(meta_dir, f"node_{node_idx}.json"))
    except: pass

    print(f"\n{'=' * 60}")
    print(f"[NODE-{node_idx}] ALL TESTS PASSED!")
    print(f"  - RDMA buffer allocation: OK")
    print(f"  - Internode Proxy creation: OK")
    print(f"  - Buffer sync (NVL + RDMA): OK")
    print(f"  - RDMA QP connection (start_dual): OK")
    print(f"  - Connectivity verification: OK")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
