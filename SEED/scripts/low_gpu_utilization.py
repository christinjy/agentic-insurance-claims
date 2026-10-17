import torch
import os
import sys
import time
import argparse
import subprocess
import threading


def get_visible_gpu_indices():
    """
    返回当前进程实际可见的物理 GPU 编号列表。
    集群通过 CUDA_VISIBLE_DEVICES 做隔离，必须用这个才能查对应的物理卡。
    若未设置 CUDA_VISIBLE_DEVICES，则返回 [0, 1, ..., gpu_number-1]。
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "NoDevFiles":
        return [int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()]
    # 未设置时用 torch 探测
    n = torch.cuda.device_count()
    return list(range(n))


def gpu_utilization(physical_gpu_id):
    """用 nvidia-smi 查询指定物理 GPU 的当前利用率（%）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits",
             f"--id={physical_gpu_id}"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out)
    except Exception:
        return 0


def keep_gpu_busy(logical_i, phys_i, threshold, stop_event):
    """
    每张卡独立一个线程，持续监控+补偿。
    - 利用率 < threshold 时立即开始矩阵乘法补偿
    - 利用率 >= threshold 时短暂休眠后继续检测
    - 补偿期间检测频率高（每 10 次矩阵乘法查一次），避免过补偿
    """
    device = torch.device(f"cuda:{logical_i}")
    # A100 80GB: 8192×8192 float16 矩阵乘法能有效拉高利用率，显存占用 ~0.5GB
    sz = 8192
    a = torch.rand(sz, sz, dtype=torch.float16, device=device)
    b = torch.rand(sz, sz, dtype=torch.float16, device=device)

    print(f"[low_gpu_util] 线程启动: logical={logical_i} physical={phys_i} threshold={threshold}%", flush=True)

    while not stop_event.is_set():
        util = gpu_utilization(phys_i)
        if util < threshold:
            # 利用率不足，持续做矩阵乘法补偿
            count = 0
            while not stop_event.is_set():
                # 连续做 10 次再查一次利用率（减少 nvidia-smi 开销）
                for _ in range(10):
                    a = a @ b
                count += 1
                if count % 3 == 0:
                    torch.cuda.synchronize(logical_i)
                    cur_util = gpu_utilization(phys_i)
                    if cur_util >= threshold:
                        print(f"[low_gpu_util] GPU{phys_i} 利用率恢复: {cur_util}% >= {threshold}%", flush=True)
                        break
        else:
            # 利用率已达标，短暂休眠（0.5s）后再检测
            # 注意：不能 sleep 太长，否则训练阶段空闲 2s+ 会被漏掉
            time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_number", required=False, default=1, type=int)
    parser.add_argument("--gpu_utilization_threshold", required=False, default=70, type=int)
    args = parser.parse_args()

    # 获取实际物理 GPU 编号（解决集群 CUDA_VISIBLE_DEVICES 隔离问题）
    physical_ids = get_visible_gpu_indices()
    # 只取前 gpu_number 个
    physical_ids = physical_ids[:args.gpu_number]
    print(f"[low_gpu_util] 监控物理 GPU: {physical_ids}, 阈值={args.gpu_utilization_threshold}%", flush=True)

    stop_event = threading.Event()

    # ★ 每张卡独立一个线程，并行监控+补偿（原版串行会导致某些卡持续低利用率）
    threads = []
    for logical_i, phys_i in enumerate(physical_ids):
        t = threading.Thread(
            target=keep_gpu_busy,
            args=(logical_i, phys_i, args.gpu_utilization_threshold, stop_event),
            daemon=True,
            name=f"gpu-keeper-{phys_i}",
        )
        t.start()
        threads.append(t)

    try:
        # 主线程永久等待，被 kill 时 daemon 线程自动退出
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[low_gpu_util] 收到中断，停止所有守护线程", flush=True)
        stop_event.set()
        for t in threads:
            t.join(timeout=3)
