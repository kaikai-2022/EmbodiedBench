#!/usr/bin/env python3
"""
一键 ACT 模型训练脚本:自动检测空闲 GPU -> 多卡生成轨迹 -> 单卡训练

功能:
  - 自动检测指定 GPU 列表中的空闲 GPU(显存<阈值 & util<10%)
  - 轨迹生成:使用所有空闲 GPU 并行
  - 格式转换 + 单卡训练:从空闲 GPU 中选 1 张(ID 最小)
  - 实时进度输出(轨迹生成数量、转换进度、训练 loss)

用法:
    python scripts/model_train/train_act.py \
        --task place_flask_on_the_mat \
        --num 200 \
        --available-gpus 2,3,4,5,6,7 \
        --train-steps 100000 \
        --use-in-memory
"""

import argparse
import os
import subprocess
import threading
import time
import sys
import re
from datetime import datetime

PROJECT_ROOT = "/ssd/qinmaokai/workspace/SciVLABench"
LEROBOT_ROOT = "/ssd/qinmaokai/workspace/lerobot"
CONDA_SH = "/ssd/qinmaokai/miniconda3_new/etc/profile.d/conda.sh"


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{timestamp()}] {msg}", flush=True)


def conda_run(env_name, cmd):
    """返回在指定 conda 环境运行的 shell 命令"""
    if env_name == "vlabench_2":
        return f"source {CONDA_SH} && conda activate {env_name} && {cmd}"
    return f"source {CONDA_SH} && conda activate {env_name} && {cmd}"


def detect_free_gpus(gpu_ids, max_mem_mb=4096):
    """检测指定的 GPU ID 列表中哪些真正空闲。

    判定标准:
      - 当前显存占用 < max_mem_mb (默认 4GB,允许其他小任务驻留)
      - GPU 进程数 == 0(或只有 CUDA context 但无活跃计算)

    Returns:
        list[int]: 真正空闲的 GPU ID 列表(按 ID 升序)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpu_stats = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            idx, mem, util = int(parts[0]), int(parts[1]), int(parts[2])
            gpu_stats[idx] = {"mem": mem, "util": util}

        free = []
        busy = []
        for gid in gpu_ids:
            stats = gpu_stats.get(gid)
            if stats is None:
                busy.append((gid, "未检测到"))
                continue
            # 真正空闲: 显存占用小且 GPU 利用率低
            if stats["mem"] < max_mem_mb and stats["util"] < 10:
                free.append(gid)
            else:
                busy.append((gid, f"显存={stats['mem']}MB, util={stats['util']}%"))
        return free, busy
    except Exception as e:
        log(f"⚠️ GPU 检测失败: {e}")
        return [], [(g, "检测失败") for g in gpu_ids]


class ProgressMonitor:
    """通用进度监控器"""
    def __init__(self, name, check_fn, interval=5):
        self.name = name
        self.check_fn = check_fn
        self.interval = interval
        self.running = False
        self.thread = None

    def _loop(self):
        while self.running:
            self.check_fn()
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        log(f"[{self.name}] 监控已启动")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        log(f"[{self.name}] 监控已停止")


class TrajectoryGenMonitor(ProgressMonitor):
    """轨迹生成监控"""
    def __init__(self, task_name, series_name, num_target, interval=10):
        self.task_name = task_name
        self.series_name = series_name
        self.num_target = num_target
        self.start_time = None
        super().__init__("轨迹生成", self._check)

    def _check(self):
        # 实际保存路径是 series_name/task_name/（与 generate_trajectories.sh 一致）
        task_dir = f"{PROJECT_ROOT}/dataset/training_data/{self.series_name}/{self.task_name}"
        if not os.path.exists(task_dir):
            return
        files = [f for f in os.listdir(task_dir) if f.endswith(".hdf5")]
        count = len(files)
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = count / elapsed if elapsed > 0 else 0
        remaining = self.num_target - count
        eta = remaining / rate / 60 if rate > 0 else None
        if isinstance(eta, float) and eta > 0:
            log(f"[{self.name}] 已生成 {count}/{self.num_target} 轨迹 (速率 {rate:.1f}/s) 剩余约 {eta:.1f} 分钟")
        else:
            log(f"[{self.name}] 已生成 {count}/{self.num_target} 轨迹")

    def start(self):
        self.start_time = time.time()
        super().start()


class ACTTrainMonitor(ProgressMonitor):
    """ACT 训练监控"""
    def __init__(self, log_file, total_steps, interval=15):
        self.log_file = log_file
        self.total_steps = total_steps
        self.last_loss = None
        self.last_step = 0
        super().__init__("训练", self._check)

    def _check(self):
        if not os.path.exists(self.log_file):
            return
        try:
            with open(self.log_file) as f:
                content = f.read()
            lines = content.split("\n")

            # 查找最新的训练日志行（lerobot 格式）
            for line in reversed(lines):
                if "step:" in line and "loss:" in line:
                    # 提取 step 和 loss
                    m_step = re.search(r"step:(\d+)", line)
                    m_loss = re.search(r"loss:([0-9.]+)", line)
                    m_grad = re.search(r"grdn:([0-9.]+)", line)

                    if m_step:
                        step = int(m_step.group(1))
                        if step != self.last_step:
                            self.last_step = step
                            pct = step / self.total_steps * 100
                            log_msg = f"[训练] Step {step}/{self.total_steps} ({pct:.1f}%)"

                            if m_loss:
                                loss = float(m_loss.group(1))
                                log_msg += f" | Loss={loss:.6f}"

                            if m_grad:
                                grad = float(m_grad.group(1))
                                log_msg += f" | Grad={grad:.2f}"

                            log(log_msg)
                    break
        except:
            pass


def print_conv_progress(log_file, target_num=200):
    """打印转换进度（统计文件数，不是帧数）"""
    if not os.path.exists(log_file):
        return
    try:
        with open(log_file) as f:
            content = f.read()
        lines = content.split("\n")

        # 统计文件数（不是帧数）
        # "Creating parquet" 100% 表示一个 episode 的 parquet 文件创建完成
        parquet_files = 0
        for line in lines:
            if "Creating parquet" in line and "100%" in line:
                parquet_files += 1

        # "Generating train split" 行表示 parquet 转为 dataset 的进度
        # 找最后一行的 "Generating train split: X examples" 来获取已处理的 frame 数（用于判断进度）
        frames_processed = 0
        for line in lines:
            if "Generating train split:" in line and "examples" in line:
                m = re.search(r"Generating train split:\s*(\d+)\s*examples", line)
                if m:
                    frames_processed = int(m.group(1))

        # 统计跳过/失败的
        skipped = 0
        for line in lines:
            if ("Skipping" in line or "skipping" in line or "Empty" in line) and "%" not in line:
                skipped += 1

        # 检查是否完成（"Done" 出现在转换日志中表示完成）
        if "Done" in content or "consolidate" in content or frames_processed > 0:
            log(f"  [转换] ✅ 完成! 已处理 {frames_processed} 帧 ({parquet_files} 个文件)")
            return

        # 显示进度摘要（基于 parquet 文件数和帧数）
        if parquet_files > 0:
            log(f"  [转换] 已处理 {frames_processed} 帧 | 文件: {parquet_files} | 跳过: {skipped}")

        # 打印关键行
        for line in reversed(lines[-20:]):
            stripped = line.strip()
            if stripped and any(k in stripped for k in ["ERROR", "WARNING", "Empty file"]):
                log(f"  [转换] ⚠️ {stripped[:100]}")
                break
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="一键 ACT 模型训练脚本")
    parser.add_argument("--task", required=True, help="任务名（不带 _series 后缀）")
    parser.add_argument("--num", type=int, required=True, help="目标轨迹数量")
    parser.add_argument("--gpus", default="2", help="GPU 列表（单卡默认用 2 号卡）")
    parser.add_argument("--train-steps", type=int, default=30000, help="训练步数（老师推荐: 单卡 A800 30k 步 1-2h）")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size（LabUtopia 用 128，过大会触发 PIL decode 瓶颈）")
    parser.add_argument("--num-workers", type=int, default=8, help="数据加载线程数（LabUtopia 验证甜点为 8）")
    parser.add_argument("--save-freq", type=int, default=5000, help="保存 checkpoint 频率")
    parser.add_argument("--samples-per-gpu", type=int, default=40, help="每张 GPU 每轮最大尝试次数（默认 40）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率（老师推荐: 1e-4 for ACT）")
    parser.add_argument("--use-amp", action="store_true", default=True, help="开启 AMP 混合精度加速（默认开启）")
    parser.add_argument("--resolution", type=int, default=256, help="图像分辨率（需先用 convert_to_lerobot_act.py --resolution 生成对应数据集）")
    parser.add_argument("--freeze-encoder", action="store_true", default=False, help="冻结 ResNet18 视觉编码器（论文默认行为，节省约 30%% 计算量）")
    parser.add_argument("--use-ema", action="store_true", default=True, help="启用 EMA 模型（提升收敛速度和稳定性）")
    parser.add_argument("--use-in-memory", action="store_true", default=False, help="使用 in_memory 预加载数据集（实验性，需要先生成并切到 256 分辨率）")
    parser.add_argument("--skip-gen", action="store_true", help="跳过轨迹生成")
    parser.add_argument("--skip-conv", action="store_true", help="跳过格式转换")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练")
    parser.add_argument("--available-gpus", default="2,3,4,5,6,7",
                        help="可用 GPU 列表(逗号分隔),脚本会检测真实空闲的卡,空闲卡用于轨迹生成,选择 1 张用于训练")
    parser.add_argument("--max-mem-mb", type=int, default=4096,
                        help="GPU 空闲判定阈值:显存占用低于此值(MB)且 util<10%% 视为空闲")
    args = parser.parse_args()

    series_name = f"{args.task}_series"
    # 在 repo_id 里嵌入分辨率,避免不同分辨率数据集冲突
    repo_id = f"vlabench_{args.task}_{args.num}ep_r{args.resolution}"
    train_log = f"/tmp/train_act_{args.task}_r{args.resolution}.log"

    # LeRobot cache 路径
    lerobot_cache = f"{os.path.expanduser('~')}/.cache/huggingface/lerobot/{repo_id}"

    log("=" * 60)
    log("ACT 模型一键训练流程启动")
    log("=" * 60)
    log(f"  任务: {args.task}")
    log(f"  轨迹目标: {args.num}")
    log(f"  GPUs: {args.gpus}")
    log(f"  训练步数: {args.train_steps}")
    log("=" * 60)

    # ========== 0. GPU 空闲检测 ==========
    requested_gpus = [int(x.strip()) for x in args.available_gpus.split(",")]
    log(f"[0/4] 检测 GPU 空闲状态(候选: {requested_gpus}, 阈值: {args.max_mem_mb}MB)...")
    free_gpus, busy_gpus = detect_free_gpus(requested_gpus, args.max_mem_mb)

    log(f"  ✓ 空闲 GPU ({len(free_gpus)} 张): {free_gpus if free_gpus else '无'}")
    if busy_gpus:
        for gid, reason in busy_gpus:
            log(f"  ✗ GPU {gid} 占用: {reason}")

    if not free_gpus:
        log("ERROR: 没有空闲的 GPU 可用,无法继续")
        sys.exit(1)

    # 训练用 GPU = 空闲 GPU 中 ID 最小的(避免抢占重要任务)
    train_gpu = free_gpus[0]
    # 轨迹生成用 GPU = 全部空闲 GPU(并行)
    gen_gpus = free_gpus

    log(f"  → 轨迹生成将使用 {len(gen_gpus)} 张卡: {gen_gpus}")
    log(f"  → 单卡训练将使用 GPU {train_gpu}")
    log("=" * 60)

    # ========== 1. 轨迹生成 ==========
    if not args.skip_gen:
        log("[1/4] 启动多卡并行轨迹生成...")

        gen_gpus_str = ",".join(str(g) for g in gen_gpus)
        cmd = conda_run("vlabench_2", f"""
bash {PROJECT_ROOT}/scripts/model_train/generate_trajectories.sh \\
    --task {args.task} \\
    --num {args.num} \\
    --gpus {gen_gpus_str} \\
    --samples {args.samples_per_gpu}
        """)

        # 后台运行生成
        proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

        # 监控线程
        monitor = TrajectoryGenMonitor(args.task, series_name, args.num, interval=15)
        monitor.start()

        # 实时打印输出（只打印关键行）
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="ignore").rstrip()
            if decoded and any(k in decoded for k in ["GPU", "生成", "条", "进度", "Episode", "success", "完成", "失败", "Error", "ERROR", "多卡", "并行"]):
                log(f"  {decoded[:120]}")

        proc.wait()
        monitor.stop()
        log(f"[1/4] 轨迹生成完成，退出码: {proc.returncode}")
        if proc.returncode != 0:
            log("ERROR: 轨迹生成失败")
            sys.exit(1)
    else:
        log("[1/4] 跳过 (--skip-gen)")

    # ========== 2. 格式转换 ==========
    conv_log = f"/tmp/conv_act_{args.task}.log"

    if not args.skip_conv:
        log("[2/4] 启动 LeRobot 格式转换...")

        # 先删除旧 cache
        if os.path.exists(lerobot_cache):
            log(f"  删除旧 LeRobot 数据: {lerobot_cache}")
            subprocess.run(f"rm -rf {lerobot_cache}", shell=True)

        cmd = conda_run("vlabench_2", f"""
export PYTHONPATH={LEROBOT_ROOT}:{PROJECT_ROOT} && \\
export HF_HOME={PROJECT_ROOT}/.cache/huggingface && \\
export HF_HUB_OFFLINE=1 && \\
export HF_DATASETS_OFFLINE=1 && \\
cd {PROJECT_ROOT} && \\
python scripts/convert_to_lerobot_act.py \\
    --dataset-name {repo_id} \\
    --dataset-path {PROJECT_ROOT}/dataset/training_data \\
    --task-list {series_name} \\
    --max-files {args.num} \\
    --resolution {args.resolution} \\
    > {conv_log} 2>&1
        """)

        # 实时打印转换输出
        proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
        monitor2 = ProgressMonitor("格式转换", lambda: print_conv_progress(conv_log, args.num))
        monitor2.interval = 10
        monitor2.start()
        proc.wait()
        monitor2.stop()

        log(f"[2/4] 转换完成，退出码: {proc.returncode}")
        if proc.returncode != 0:
            with open(conv_log) as f:
                log(f"转换日志最后 20 行:\n" + "\n".join(f.readlines()[-20:]))
            sys.exit(1)
    else:
        log("[2/4] 跳过 (--skip-conv)")

    # ========== 3. 单卡训练 ==========
    if not args.skip_train:
        log(f"[3/4] 启动单卡训练 (GPU {train_gpu},选自空闲 GPU 列表)...")
        log("=" * 60)

        # 单卡训练:batch_size 就是全局 batch_size
        num_gpus = 1

        # ⚠️ LeRobot 会自动创建输出目录,不允许预先存在
        # 如果目录已存在 torchrun 会报 FileExistsError
        # 解决方案: 每次运行用不同时间戳(已实现);如需重跑同名目录,手动 rm -rf

        cmd = conda_run("vlabench_2", f"""
export PYTHONPATH={LEROBOT_ROOT}:{PROJECT_ROOT} && \\
export HF_HOME={PROJECT_ROOT}/.cache/huggingface && \\
export HF_HUB_OFFLINE=1 && \\
export HF_DATASETS_OFFLINE=1 && \\
export CUDA_VISIBLE_DEVICES={train_gpu} && \\
cd {LEROBOT_ROOT} && \\
python -m torch.distributed.run --standalone --nproc_per_node={num_gpus} lerobot/scripts/train.py \\
    --policy.type=act \\
    --policy.optimizer_lr={args.lr} \\
    --policy.freeze_backbone={str(args.freeze_encoder).lower()} \\
    --dataset.repo_id={repo_id} \\
    --dataset.local_files_only=true \\
    --dataset.image_transforms.enable=false \\
    --dataset.use_imagenet_stats=true \\
    --dataset.use_in_memory={str(args.use_in_memory).lower()} \\
    --batch_size={args.batch_size} \\
    --num_workers={args.num_workers} \\
    --offline.steps={args.train_steps} \\
    --save_freq={args.save_freq} \\
    --eval_freq=1000000 \\
    --log_freq=10 \\
    --device=cuda \\
    --seed=1000 \\
    --use_amp=false \
    > {train_log} 2>&1
        """)

        proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")

        monitor3 = ACTTrainMonitor(train_log, args.train_steps)
        monitor3.start()

        log(f"[3/4] 训练 PID={proc.pid}，日志: {train_log}")
        log(f"  GPU: {train_gpu} (单卡,自动从空闲 GPU 中选最小 ID)")
        log(f"  batch_size: {args.batch_size}")
        log("  提示: tail -f " + train_log + " 查看实时日志")

        proc.wait()
        monitor3.stop()

        log(f"[3/4] 训练进程退出，退出码: {proc.returncode}")
        log("=" * 60)

        if proc.returncode == 0:
            # 显示生成的 checkpoint (在 LeRobot 默认输出目录中找最新 run)
            try:
                import glob as glob_mod
                ckpt_dirs = sorted(glob_mod.glob(f"{LEROBOT_ROOT}/outputs/train/*/checkpoints/"), key=os.path.getmtime, reverse=True)
                if ckpt_dirs:
                    latest_ckpt = ckpt_dirs[0]
                    ckpts = subprocess.run(f"ls -la {latest_ckpt} 2>/dev/null | tail -10",
                        shell=True, capture_output=True, text=True)
                    if ckpts.stdout:
                        log(f"生成的 checkpoints (最新 run: {latest_ckpt}):")
                        for line in ckpts.stdout.strip().split("\n")[-5:]:
                            if line.strip():
                                log(f"  {line}")
            except:
                pass
    else:
        log("[3/4] 跳过 (--skip-train)")

    # ========== 4. 完成 ==========
    log("[4/4] 全部完成!")
    log("=" * 60)
    log(f"训练日志: {train_log}")
    log(f"模型输出目录: lerobot/outputs/train/ (LeRobot 自动生成)")

    if not args.skip_train:
        log(f"\n下一步:")
        log(f"  1. 修复 checkpoint 的 config.json（添加 \"type\": \"act\"）")
        log(f"  2. 运行评估前先找到最新 checkpoint 目录: ls lerobot/outputs/train/ | tail -1")
        log(f"  3. 评估: python scripts/evaluate_policy.py --policy act --model_ckpt <checkpoint_dir>/pretrained_model")

    log("=" * 60)


if __name__ == "__main__":
    main()
