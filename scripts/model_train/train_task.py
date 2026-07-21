#!/usr/bin/env python3
"""
一键训练脚本：生成轨迹 -> 转换为 LeRobot 格式 -> 计算 norm_stats -> 添加 config -> 开始训练

功能：
  - 实时进度输出（轨迹生成数量、转换进度、训练 loss）
  - 各阶段独立线程监控，日志实时打印

用法:
    python scripts/model_train/train_task.py --task pick_cylinder_mid_pour_object --num 200 --gpus 0,1,2,3,4,5,6 --train-steps 100000
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
OPENPI_DIR = f"{PROJECT_ROOT}/third_party/openpi"
CONDA_SH = "/opt/miniconda3/etc/profile.d/conda.sh"


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{timestamp()}] {msg}", flush=True)


def conda_run(env_name, cmd):
    """返回在指定 conda 环境运行的 shell 命令"""
    if env_name == "openpi":
        # 使用 openpi 自带的 venv
        venv_dir = f"{OPENPI_DIR}/examples/vlabench/.venv"
        return f"source {venv_dir}/bin/activate && {cmd}"
    return f"source {CONDA_SH} && conda activate {env_name} && {cmd}"


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


class NormStatsMonitor(ProgressMonitor):
    """norm_stats 计算监控"""
    def __init__(self, log_file, interval=30):
        self.log_file = log_file
        self.last_progress = -1
        super().__init__("norm_stats", self._check)

    def _check(self):
        if not os.path.exists(self.log_file):
            return
        try:
            with open(self.log_file) as f:
                content = f.read()
            lines = content.split("\n")
            for line in reversed(lines):
                # 查找进度信息
                if "Computing stats" in line:
                    log(f"[{self.name}] {line.strip()[:100]}")
                    return
                if "Done" in line or "Saved" in line:
                    log(f"[{self.name}] {line.strip()[:100]}")
                    return
        except:
            pass


class TrainMonitor(ProgressMonitor):
    """训练监控"""
    def __init__(self, log_file, interval=15):
        self.log_file = log_file
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

            # 最新进度行
            progress_line = ""
            for line in reversed(lines):
                if "Progress on" in line:
                    progress_line = line
                    break

            if progress_line:
                # 提取 step 和 loss
                m = re.search(r"(\d+\.?\d*)it/(\d+)kit", progress_line)
                if m:
                    cur = float(m.group(1).rstrip("k"))
                    total = int(m.group(2)) * 1000
                    pct = cur / total * 100
                    log(f"[训练] Step {cur:.0f}/{total} ({pct:.1f}%)")

            # 最新 loss 行
            for line in reversed(lines):
                if "Step" in line and "loss=" in line:
                    m = re.search(r"loss=([0-9.]+)", line)
                    if m:
                        loss = float(m.group(1))
                        if loss != self.last_loss:
                            self.last_loss = loss
                            log(f"[训练] Loss={loss:.6f}")
                    break
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="一键训练脚本")
    parser.add_argument("--task", required=True, help="任务名（不带 _series 后缀）")
    parser.add_argument("--num", type=int, required=True, help="目标轨迹数量")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6", help="GPU 列表")
    parser.add_argument("--train-steps", type=int, default=100000, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=7, help="batch size")
    parser.add_argument("--samples-per-gpu", type=int, default=40, help="每张 GPU 每轮最大尝试次数")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--skip-conv", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    args = parser.parse_args()

    series_name = f"{args.task}_series"
    lerobot_dataset = series_name
    train_config_name = f"pi05_ft_{args.task}"
    train_log = f"/tmp/train_{args.task}.log"

    log("=" * 60)
    log("一键训练流程启动")
    log("=" * 60)
    log(f"  任务: {args.task}")
    log(f"  轨迹目标: {args.num}")
    log(f"  GPUs: {args.gpus}")
    log(f"  训练步数: {args.train_steps}")
    log("=" * 60)

    # ========== 1. 轨迹生成 ==========
    if not args.skip_gen:
        log("[1/5] 启动轨迹生成...")
        gen_script = f"{PROJECT_ROOT}/scripts/model_train/generate_trajectories.sh"
        cmd = conda_run("vlabench_2", f"bash {gen_script} --task {args.task} --num {args.num} --gpus {args.gpus} --samples {args.samples_per_gpu}")

        # 后台运行生成
        proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)

        # 监控线程
        monitor = TrajectoryGenMonitor(args.task, series_name, args.num, interval=15)
        monitor.start()

        # 实时打印输出（只打印关键行）
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="ignore").rstrip()
            if decoded and any(k in decoded for k in ["GPU", "生成", "条", "进度", "Episode", "success", "完成", "失败", "Error", "ERROR"]):
                log(f"  {decoded[:120]}")

        proc.wait()
        monitor.stop()
        log(f"[1/5] 轨迹生成完成，退出码: {proc.returncode}")
        if proc.returncode != 0:
            log("ERROR: 轨迹生成失败")
            sys.exit(1)
    else:
        log("[1/5] 跳过 (--skip-gen)")

    # ========== 2. 格式转换 ==========
    lerobot_cache = f"{PROJECT_ROOT}/.cache/huggingface/lerobot/{lerobot_dataset}"
    conv_log = f"/tmp/conv_{args.task}.log"

    if not args.skip_conv:
        if os.path.exists(lerobot_cache):
            log(f"[2/5] 删除旧 LeRobot 数据: {lerobot_cache}")
            subprocess.run(f"rm -rf {lerobot_cache}", shell=True)

        log("[2/5] 启动 LeRobot 格式转换...")
        cmd = conda_run("openpi", f"""
export PYTHONPATH={OPENPI_DIR}/src:{PROJECT_ROOT} && \\
export HF_HOME={PROJECT_ROOT}/.cache/huggingface && \\
export HF_LEROBOT_HOME={PROJECT_ROOT}/.cache/huggingface/lerobot && \\
export HF_HUB_OFFLINE=1 && \\
export HF_DATASETS_OFFLINE=1 && \\
python {PROJECT_ROOT}/scripts/convert_to_lerobot.py \\
    --dataset-name {lerobot_dataset} \\
    --dataset-path {PROJECT_ROOT}/dataset/training_data \\
    --task-list {series_name} \\
    --max-files {args.num} \\
    > {conv_log} 2>&1
        """)

        # 实时打印转换输出
        proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
        monitor2 = ProgressMonitor("格式转换", lambda: print_conv_progress(conv_log, args.num))
        monitor2.interval = 10
        monitor2.start()
        proc.wait()
        monitor2.stop()
        log(f"[2/5] 转换完成，退出码: {proc.returncode}")
        if proc.returncode != 0:
            with open(conv_log) as f:
                log(f"转换日志最后 20 行:\n" + "\n".join(f.readlines()[-20:]))
            sys.exit(1)
    else:
        log("[2/5] 跳过 (--skip-conv)")

    # ========== 3. 添加 config (必须在 norm_stats 之前) ==========
    if not args.skip_config:
        log("[3/5] 添加训练配置...")
        add_config(args.task, series_name, train_config_name, args.train_steps, args.batch_size)
    else:
        log("[3/5] 跳过 (--skip-config)")

    # ========== 4. norm_stats ==========
    norm_log = f"/tmp/norm_stats_{args.task}.log"

    log("[4/5] 启动 norm_stats 计算...")
    cmd = conda_run("openpi", f"""
export PYTHONPATH={OPENPI_DIR}/src:{PROJECT_ROOT} && \\
export HF_HOME={PROJECT_ROOT}/.cache/huggingface && \\
export HF_LEROBOT_HOME={PROJECT_ROOT}/.cache/huggingface/lerobot && \\
export HF_HUB_OFFLINE=1 && \\
export HF_DATASETS_OFFLINE=1 && \\
export TRANSFORMERS_OFFLINE=1 && \\
export CUDA_VISIBLE_DEVICES=0 && \\
python {OPENPI_DIR}/scripts/compute_norm_stats.py --config-name={train_config_name} \\
    > {norm_log} 2>&1
        """)
    proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
    monitor3 = NormStatsMonitor(norm_log)
    monitor3.start()
    proc.wait()
    monitor3.stop()
    log(f"[4/5] norm_stats 完成，退出码: {proc.returncode}")
    if proc.returncode != 0:
        with open(norm_log) as f:
            log(f"norm_stats 日志最后 20 行:\n" + "\n".join(f.readlines()[-20:]))
        sys.exit(1)

    # ========== 5. 训练 ==========
    log("[5/5] 启动训练...")
    log("=" * 60)
    cmd = conda_run("openpi", f"""
export PYTHONPATH={OPENPI_DIR}/src:{PROJECT_ROOT} && \\
export HF_HOME={PROJECT_ROOT}/.cache/huggingface && \\
export HF_LEROBOT_HOME={PROJECT_ROOT}/.cache/huggingface/lerobot && \\
export HF_HUB_OFFLINE=1 && \\
export HF_DATASETS_OFFLINE=1 && \\
export TRANSFORMERS_OFFLINE=1 && \\
export WANDB_API_KEY="" && \\
export WANDB_MODE=offline && \\
export CUDA_VISIBLE_DEVICES={args.gpus} && \\
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 && \\
python {OPENPI_DIR}/scripts/train.py {train_config_name} \\
    --exp-name={train_config_name}_{args.train_steps}steps \\
    --assets_base_dir={PROJECT_ROOT}/.cache/openpi/assets \\
    --checkpoint_base_dir=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/checkpoints \\
    --num_train_steps={args.train_steps} \\
    --batch_size={args.batch_size} \\
    --num_workers=32 \\
    --log_interval=100 \\
    --save_interval=5000 \\
    --keep_period=10000 \\
    --overwrite \\
    > {train_log} 2>&1
        """)
    proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")

    monitor4 = TrainMonitor(train_log)
    monitor4.start()

    log(f"[5/5] 训练 PID={proc.pid}，日志: {train_log}")
    log("  提示: grep 'Step' {train_log} 查看 loss")
    log("  提示: grep 'Progress on' {train_log} 查看进度")

    proc.wait()
    monitor4.stop()
    log(f"[5/5] 训练进程退出，退出码: {proc.returncode}")
    log("=" * 60)
    log("全部完成!")
    log("=" * 60)


def add_config(task_name, series_name, config_name, train_steps, batch_size):
    """添加 config 到 openpi 配置文件"""
    config_file = f"{OPENPI_DIR}/src/openpi/training/config.py"
    with open(config_file) as f:
        content = f.read()

    if f'name="{config_name}"' in content:
        log(f"  配置 {config_name} 已存在，跳过")
    else:
        new_cfg = f'''
    # {task_name}
    TrainConfig(
        name="{config_name}",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotVLABenchDataConfig(
            repo_id="{series_name}",
            base_config=DataConfig(local_files_only=True, prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/ssd/qinmaokai/.cache/openpi/vlabench_checkpoints/pi05-primitive-10task/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        assets_base_dir="/ssd/qinmaokai/workspace/SciVLABench/.cache/openpi/assets",
        checkpoint_base_dir="/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/checkpoints",
        ema_decay=None,
        num_train_steps={train_steps},
        batch_size={batch_size},
        num_workers=32,
        log_interval=100,
        save_interval=5000,
        keep_period=10000,
    ),
'''
        # 在 place_beaker_on_the_mat 之前插入
        marker = '    # place_beaker_on_the_mat: pick beaker and place on mat (simpler task)'
        if marker in content:
            content = content.replace(marker, new_cfg.strip() + '\n' + marker)
            with open(config_file, "w") as f:
                f.write(content)
            log(f"  已添加配置 {config_name}")
        else:
            log(f"  警告: 未找到插入标记，手动添加 {config_name} 到 {config_file}")

    # 添加任务注册
    init_file = f"{PROJECT_ROOT}/VLABench/configs/__init__.py"
    with open(init_file) as f:
        init_content = f.read()
    if f'"{series_name}"' in init_content:
        log(f"  任务注册已存在，跳过")
    else:
        marker2 = '    "place_beaker_on_the_mat_series":["place_beaker_on_the_mat"],'
        new_entry = f'    "{series_name}":["{task_name}"],\n{marker2}'
        init_content = init_content.replace(marker2, new_entry)
        with open(init_file, "w") as f:
            f.write(init_content)
        log(f"  已添加任务注册 {series_name}")


def print_conv_progress(log_file, target_num=200):
    """打印转换进度"""
    if not os.path.exists(log_file):
        return
    try:
        with open(log_file) as f:
            content = f.read()
        lines = content.split("\n")

        # 统计进度
        processed = processed_files = skipped = 0
        for line in lines:
            if "Creating parquet" in line and "100%" in line:
                processed += 1
            if "Skipping" in line or "skipping" in line or "Empty" in line:
                skipped += 1
            if "Map:" in line and "%" in line:
                # 从 Map 行提取进度
                import re
                m = re.search(r"(\d+)/(\d+)", line)
                if m:
                    processed_files = int(m.group(1))

        # 检查是否完成
        if "100%" in content and ("Done" in content or "Writing" in content):
            log(f"  [转换] ✅ 完成! 处理了 {processed} 个文件")
            return

        # 显示进度摘要
        remaining = max(0, target_num - processed)
        if processed > 0:
            log(f"  [转换] 已处理: {processed} | 成功: {processed - skipped} | 跳过: {skipped} | 剩余: {remaining}")

        # 打印关键行
        for line in reversed(lines[-20:]):
            stripped = line.strip()
            if stripped and any(k in stripped for k in ["ERROR", "WARNING", "Skipping", "Empty file"]):
                log(f"  [转换] ⚠️ {stripped[:100]}")
                break
    except:
        pass


if __name__ == "__main__":
    main()
