# OpenPI Policy Evaluation Guide (Current)

> 当前评测流程说明，基于 `scripts/evaluate_openpi.py`（argparse 风格，独立 `run_episode` 循环，动态加载 `autogen_tasks`）。
>
> 旧版基于 `third_party/openpi/examples/vlabench/eval.py`（tyro + Evaluator 框架）的指南已归档在 `OpenPI_Evaluation_Guide.md` 和 `OpenPI_Evaluation_Technical_Guide.md`，仅作历史参考。

---

## 目录

1. [系统架构](#1-系统架构client-server-模式)
2. [环境准备](#2-环境准备)
3. [启动服务器（终端1）](#3-启动服务器终端1)
4. [运行评测（终端2）](#4-运行评测终端2)
5. [CLI 参数](#5-cli-参数)
6. [autogen_tasks 工作流](#6-autogen_tasks-工作流)
7. [完整调用链路](#7-完整调用链路)
8. [OpenPI 模型推理流程](#8-openpi-模型推理流程)
9. [成功条件系统](#9-成功条件系统)
10. [输出与视频](#10-输出与视频)
11. [常见问题与排错](#11-常见问题与排错)
12. [快速命令参考](#12-快速命令参考)

---

## 1. 系统架构（Client-Server 模式）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Terminal 1: OpenPI Policy Server (OpenPI .venv)                         │
│                                                                         │
│  scripts/serve_policy.py                                                │
│      ├── 加载 pi0-fast 模型权重 (JAX/Flax)                              │
│      ├── 加载 PaliGemma tokenizer                                        │
│      └── 启动 WebSocket 服务器, 监听 8000 端口                          │
│                                                                         │
│  接收: policy_input = {image, second_image, wrist_image, state, prompt} │
│  返回: {"actions": [...]}                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ WebSocket (msgpack序列化)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Terminal 2: VLABench Evaluator Client (vlabench_2 conda)               │
│                                                                         │
│  scripts/evaluate_openpi.py                                             │
│      ├── Pi0: WebSocket客户端, 封装predict()调用                         │
│      ├── run_episode(): 评测循环, 管理环境, 计算指标                    │
│      ├── VLABench环境 (MuJoCo): 仿真执行, 渲染图像, 条件判断            │
│      └── autogen_tasks 动态加载 (glob + importlib)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**为什么分两个终端/两个环境？**
- **服务器端**需要 OpenPI 的依赖（JAX、Flax 等），需要在 GPU 上运行
- **客户端**需要 VLABench 的依赖（MuJoCo、dm_control 等），管理仿真环境
- 两者通过 **WebSocket** 解耦，通过 msgpack 序列化传输数据

---

## 2. 环境准备

### 2.1 两个独立环境

#### 环境 1：OpenPI 虚拟环境（`.venv`）

位置：`third_party/openpi/examples/vlabench/.venv`

```bash
cd /ssd/mkqin/workspace/VLABench/third_party/openpi
/ssd/mkqin/miniconda3/bin/uv venv --python 3.11 examples/vlabench/.venv
source examples/vlabench/.venv/bin/activate
/ssd/mkqin/miniconda3/bin/uv pip sync examples/vlabench/requirements.txt
/ssd/mkqin/miniconda3/bin/uv pip install -e packages/openpi-client
/ssd/mkqin/miniconda3/bin/uv pip install -e /ssd/mkqin/workspace/VLABench
```

#### 环境 2：VLABench Conda 环境

使用现有的 `vlabench_2` conda 环境：

```bash
source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2
pip install openpi-client mediapy
```

### 2.2 必需文件

#### Checkpoint 文件

模型权重下载到：
```
/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task/
```

缺失时服务器首次启动会自动从 S3 下载。

#### Tokenizer 文件

PaliGemma tokenizer 路径：
```
/ssd/mkqin/.cache/openpi/big_vision/paligemma_tokenizer.model
```

如果遇到网络问题（无法访问 Google Cloud Storage），从 HuggingFace 镜像下载：
```bash
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/paligemma-3b-pt-224', 'tokenizer.model', local_dir='/tmp/tokenizer')"
cp /tmp/tokenizer/tokenizer.model /ssd/mkqin/.cache/openpi/big_vision/paligemma_tokenizer.model
mkdir -p /ssd/mkqin/.cache/openpi/big_vision
```

### 2.3 GPU 选择

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

选择至少 8GB 空闲显存的 GPU：
```bash
export CUDA_VISIBLE_DEVICES=1  # 使用 GPU 1
```

---

## 3. 启动服务器（终端1）

```bash
# 1. 激活 OpenPI 虚拟环境
source /ssd/mkqin/workspace/VLABench/third_party/openpi/examples/vlabench/.venv/bin/activate

# 2. 设置环境变量
export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

# 3. 进入 OpenPI 目录
cd /ssd/mkqin/workspace/VLABench/third_party/openpi

# 4. 启动服务器
python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pifast_ft_vlabench_primitive \
    --policy.dir=/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task
```

**预期输出**：
```
INFO:root:Loading model...
INFO:absl:Finished restoring checkpoint from /ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task/params.
INFO:root:Creating server (host: ..., ip: 127.0.1.1)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

看到 `server listening on 0.0.0.0:8000` 后服务器就绪。

### 后台启动（推荐）

```bash
nohup python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pifast_ft_vlabench_primitive \
    --policy.dir=/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task \
    > /tmp/server.log 2>&1 &
```

日志写入 `/tmp/server.log`，通过 `tail -f /tmp/server.log` 监控。

---

## 4. 运行评测（终端2）

**在另一个终端**，等待服务器就绪后：

```bash
# 1. 激活 VLABench conda 环境
source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2

# 2. 设置环境变量
export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

# 3. 进入 VLABench 项目根目录
cd /ssd/mkqin/workspace/VLABench

# 4. 运行评测
python scripts/evaluate_openpi.py \
    --tasks lift_small_beaker \
    --n_episode 1 \
    --save_dir logs/lift_small_beaker_test \
    --visualization
```

**典型输出**：
```
[DEBUG] 所有模块导入成功
[DEBUG] main() 函数开始
[DEBUG] VLABENCH_ROOT=/ssd/mkqin/workspace/VLABench/VLABench
[DEBUG] 正在连接 OpenPI server... localhost 8000
[DEBUG] OpenPI server 连接成功
[DEBUG] Pi0 agent 创建成功

============================================================
Task: lift_small_beaker (1 episodes)
============================================================
Episode 0: success=True, steps=69, conditions=[LiftCondition=OK]

--- lift_small_beaker Summary ---
  LiftCondition: 1/1 (100.0%)
  Success rate: 100.0%
  Condition score: 100.0%

Results saved to logs/lift_small_beaker_test/evaluation_result.json
```

### 评测多个任务

```bash
python scripts/evaluate_openpi.py \
    --tasks lift_beaker lift_small_beaker pick_tube_lift_tube \
    --n_episode 3 \
    --save_dir logs/multi_task_eval \
    --visualization
```

---

## 5. CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `localhost` | OpenPI server 地址 |
| `--port` | `8000` | OpenPI server 端口 |
| `--replan_steps` | `5` | 重新规划间隔（每 N 步重新调用模型推理一次） |
| `--tasks` | **必填** | 任务名列表（空格分隔），例如 `lift_beaker pick_tube_lift_tube` |
| `--n_episode` | `1` | 每个任务的 episode 数 |
| `--save_dir` | `None` | 结果和视频保存目录（**强烈建议设置**） |
| `--visualization` | `False` | 是否保存评测视频（flag，加了即启用） |
| `--max_episode_length` | `200` | 单个 episode 最大步数（可被 `task_config.json` 中的 `evaluation.max_episode_length` 覆盖） |
| `--seed` | `42` | 随机种子（每个 episode 使用 `seed + ep_i`） |

> **注意**：与旧版 `tyro.cli` 的 `--args.*` 前缀不同，新版使用 `argparse` 直接的 `--tasks`、`--port` 等。

---

## 6. autogen_tasks 工作流

`/ssd/mkqin/workspace/VLABench/VLABench/tasks/autogen_tasks/` 目录用于存放自动生成的任务。与 `VLABench/tasks/hierarchical_tasks/primitive/` 下的手写任务不同，autogen 任务由流水线生成。

### 6.1 目录结构

```
VLABench/tasks/autogen_tasks/
├── __init__.py                      # 空白，仅占位
├── base.py                          # PrimitiveTask 基类（带 intention/progress 跟踪）
├── lift_flask_series.py
├── lift_small_beaker_series.py
├── pick_tube_lift_tube_series.py
├── pick_glass_stirring_rod_insert_glass_stirring_rod_series.py
├── pick_tube_shake_tube_series.py
├── pick_small_beaker_shake_small_beaker_series.py
└── ... (其他自动生成任务)
```

`base.py` 提供的 `PrimitiveTask` 类继承自 `LM4ManipBaseTask`，实现了 `intention_score` / `progress_score` 等方法。

### 6.2 动态加载机制

`evaluate_openpi.py` 启动时会扫描 `autogen_tasks/*_series.py` 并动态导入，触发 `@register` 装饰器：

```python
# scripts/evaluate_openpi.py 第 24-32 行
import glob, importlib.util
_autogen_dir = os.path.join(
    os.environ.get("VLABENCH_ROOT", _VLABENCH_ROOT),
    "tasks", "autogen_tasks"
)
for _path in glob.glob(os.path.join(_autogen_dir, "*_series.py")):
    _name = os.path.splitext(os.path.basename(_path))[0]
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
```

**关键点**：
- 必须先 `from VLABench.tasks import *`（触发 `register` 模块初始化），再做动态导入
- autogen 任务**不需要**在 `autogen_tasks/__init__.py` 中显式导入
- `VLABench/tasks/__init__.py` 中也不需要添加 `from VLABench.tasks.autogen_tasks import *`

### 6.3 添加新 autogen 任务的清单

假设新任务名为 `lift_new_object`：

#### 步骤 1：创建任务文件

`VLABench/tasks/autogen_tasks/lift_new_object_series.py`：

```python
import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml


@register.add_config_manager("lift_new_object")
class LiftNewObjectConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="new_object_0",
            xml_path=name2class_xml["new_object"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "..."  # 实体类名
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)
        self.target_entity = "new_object_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["lift the <new_object_0>"]

    def get_condition_config(self, target_entity, **kwargs):
        self.config["task"]["conditions"] = dict(
            lift=dict(entities=["new_object_0"], lift_height=0.15)
        )


@register.add_task("lift_new_object")
class LiftNewObjectTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        return [
            partial(SkillLib.pick, target_entity_name="new_object_0"),
            partial(SkillLib.lift, lift_height=0.15),
        ]
```

#### 步骤 2：在 `name2config` 中添加映射

`VLABench/configs/__init__.py`：

```python
name2config = {
    # ... 已有映射 ...
    "lift_new_object_series": ["lift_new_object"],
}
```

> **重要**：如果忘了这一步，`BenchTaskConfigManager.__init__()` 会因为 `find_key_by_value(name2config, task_name)` 返回 `None` 而抛 `TypeError: 'NoneType' object is not iterable`。

#### 步骤 3：直接运行评测

```bash
python scripts/evaluate_openpi.py \
    --tasks lift_new_object \
    --n_episode 1 \
    --save_dir logs/lift_new_object_test \
    --visualization
```

无需修改 `evaluate_openpi.py` 本身 —— 动态加载会自动发现新任务。

---

## 7. 完整调用链路

```
python scripts/evaluate_openpi.py --tasks lift_small_beaker --n_episode 1 --save_dir logs/x --visualization
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ evaluate_openpi.py:main() (argparse 解析)                              │
│                                                                        │
│ 1. 解析 --tasks, --n_episode, --save_dir, --visualization              │
│ 2. 读取 task_config.json (获取每个任务的 max_episode_length)            │
│ 3. WebsocketClientPolicy("localhost", 8000)  →  连接 server           │
│ 4. Pi0(client, replan_steps=5)         →  封装 client                  │
│ 5. 动态加载 autogen_tasks/*_series.py  →  触发 @register 注册          │
│ 6. for task in args.tasks:                                            │
│      for ep_i in range(n_episode):                                    │
│        run_episode(env, agent, save_video=...)                        │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ evaluate_openpi.py:run_episode()                                       │
│                                                                        │
│ 1. env = load_env(task, random_init=True, eval=False, run_mode="eval") │
│ 2. env.reset()                                                         │
│ 3. condition.record_initial_state(env.physics)  ←── 记录条件初始态    │
│ 4. for i in range(max_episode_length):                                │
│      ├── obs = env.get_observation(require_pcd=False)                 │
│      ├── obs["instruction"] = env.task.get_instruction()              │
│      ├── pos, euler, gripper = Pi0.predict(obs)                        │
│      ├── action = ee_to_qpos(pos, euler) + gripper                    │
│      ├── timestep = env.step(action)                                  │
│      ├── if timestep.last(): success=True; break                      │
│      └── env.task.conditions.is_met(physics)  ←── 推进条件锁        │
│ 5. condition_results = {cond.is_met(physics) for cond in conditions}  │
│ 6. env.close()                                                         │
│ 7. return success, steps, condition_results, frames                    │
└───────────────────────────────────────────────────────────────────────┘
        │
        ├──► Pi0.predict(obs)                                          │
        │                                                           │
        │   if action_plan 为空:                                          │
        │     policy_input = {                                          │
        │       "observation/image": front_rgb,                          │
        │       "observation/second_image": second_rgb,                  │
        │       "observation/wrist_image": wrist_rgb,                    │
        │       "observation/state": [pos-offset, euler, gripper],      │
        │       "prompt": obs["instruction"]                            │
        │     }                                                         │
        │     action_chunk = model.infer(policy_input)["actions"]  ←─── WebSocket
        │     action_plan.extend(action_chunk[:5])                      │
        │   action = action_plan.popleft()                              │
        │   return target_pos, target_euler, gripper_state              │
        │                                                           │
        └──► load_env(task, ...)  ←─── 环境加载                      │
                                                                    │
            1. find_key_by_value(name2config, task) → series_name    │
            2. TASK_CONFIG[series_name] → 任务基础配置                 │
            3. robot = register.load_robot("franka")(...)              │
            4. task = register.load_task(task)(...)                    │
            5. env = LM4ManipDMEnv(task, ...)                         │
            6. env.reset() → task.build_from_config()                 │
```

### 与旧版 Evaluator 框架的对比

| 环节 | 旧版 (tyro + Evaluator) | 新版 (argparse + 内联) |
|------|------------------------|------------------------|
| CLI 解析 | `tyro.cli(Args)` + `--args.*` 前缀 | `argparse` + 直接参数 |
| 评测循环 | `Evaluator.evaluate()` → `evaluate_single_episode()` | `run_episode()` 内联 |
| 指标 | `success_rate`, `intention_score`, `progress_score` | `success_rate`, `condition_score`（按条件名） |
| 任务加载 | 静态导入 `VLABench.tasks` | 静态 + `autogen_tasks` 动态 glob |
| 视频保存 | 由 `Evaluator.save_video` 内部处理 | `mediapy.write_video` 显式调用 |

---

## 8. OpenPI 模型推理流程

```python
# scripts/evaluate_openpi.py: Pi0.predict()
def predict(self, obs, **kwargs):
    if len(self.action_plan) == 0:  # 缓冲空，重新推理
        second_image, _, image, image_wrist = obs["rgb"]
        state = obs["ee_state"]
        pos, quat, gripper_state = state[:3], state[3:7], state[-1]
        ee_euler = quaternion_to_euler(quat)
        pos -= np.array([0, -0.4, 0.78])  # 转换到模型训练时的坐标系
        state = np.concatenate([pos, ee_euler, np.array(gripper_state).reshape(-1)])
        instruction = obs["instruction"]
        policy_input = {
            "observation/image": image,
            "observation/second_image": second_image,
            "observation/wrist_image": image_wrist,
            "observation/state": state,
            "prompt": instruction,
        }
        action_chunk = self.model.infer(policy_input)["actions"]  # WebSocket
        assert len(action_chunk) >= self.replan_steps
        self.action_plan.extend(action_chunk[: self.replan_steps])

    action = self.action_plan.popleft()
    target_pos, target_euler, gripper = action[:3], action[3:6], action[-1]
    if gripper >= 0.1:
        gripper_state = np.ones(2) * 0.04
    else:
        gripper_state = np.zeros(2)
    target_pos = target_pos.copy()
    target_pos += np.array([0, -0.4, 0.78])  # 转换回 VLABench 坐标系
    return target_pos, target_euler, gripper_state
```

**坐标系转换**：模型在训练时使用 origin 偏移 `+[0, -0.4, 0.78]` 的坐标系，发送前减去，接收后加回。

**WebSocket 通信**：每次 `model.infer()` 通过 WebSocket 发送 msgpack 序列化的 `policy_input`，接收 `{"actions": [...]}`。

---

## 9. 成功条件系统

### 9.1 Condition 类（`VLABench/tasks/condition.py`）

| 条件类 | 用途 | 典型参数 |
|--------|------|----------|
| `LiftCondition` | 实体被举起到指定高度 | `entities`, `lift_height` / `target_height` |
| `ContainCondition` | 实体在容器内 | `container`, `entities` |
| `IsGraspedCondition` | 实体被机械臂抓住 | `entities`, `robot` |
| `OnCondition` | 实体在容器表面上 | `entities`, `container` |
| `ShakeCondition` | 实体被摇动 | `entity`, `robot` |
| `WaitForCondition` | 等待外部变化（颜色/溶液） | `entity`, `robot`, `change_type` |
| `PourCondition` | 实体被倾倒 | `target_entity`, `threshold` |
| `HeatedCondition` | 实体被加热源加热 | `target_entity`, `heat_source`, `duration` |

### 9.2 record_initial_state 的关键性

`LiftCondition` 等依赖初始态的条件，需要在每个 episode 第一次 `is_met()` 调用之前先调用 `record_initial_state(physics)`，否则会基于当前帧而非"初始 vs 终态"做判断。

`run_episode()` 在 episode 开始时会自动调用 `record_initial_state`：

```python
# scripts/evaluate_openpi.py: run_episode()
if hasattr(env.task, 'conditions') and env.task.conditions is not None:
    for condition in env.task.conditions.conditions:
        if hasattr(condition, 'record_initial_state'):
            condition.record_initial_state(env.physics)
```

**详细排错**：[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) 第 3 节记录了 `LiftCondition` 的 `_initial_z` 为空时误判为 True 导致死循环的 bug 及修复。

### 9.3 Condition 结果统计

每个 episode 结束后，`run_episode()` 收集每个 condition 的最终 `is_met` 结果：

```python
condition_results = {}
for cond in env.task.conditions.conditions:
    cond_name = type(cond).__name__
    condition_results[cond_name] = cond.is_met(env.physics)
```

`evaluate_openpi.py` 汇总每个 task 的 condition 通过率：

```
--- lift_small_beaker Summary ---
  LiftCondition: 1/1 (100.0%)
  IsGraspedCondition: 0/1 (0.0%)
  Success rate: 100.0%
  Condition score: 50.0%   ← 所有 condition 通过率的平均
```

---

## 10. 输出与视频

### 10.1 结果文件

```
<save_dir>/
├── evaluation_result.json       # 所有任务的结果汇总
└── <task_name>/
    └── videos/
        └── ep0_success_True_steps_69.mp4
```

`evaluation_result.json` 示例：

```json
{
  "lift_small_beaker": {
    "success_rate": 100.0,
    "condition_score": 100.0,
    "per_condition": {
      "LiftCondition": 1.0
    },
    "episodes": [
      {
        "episode": 0,
        "success": true,
        "steps": 69,
        "conditions": {
          "LiftCondition": true
        }
      }
    ]
  }
}
```

### 10.2 视频保存

需要在启动评测时加 `--visualization` flag：

```bash
python scripts/evaluate_openpi.py \
    --tasks lift_small_beaker \
    --n_episode 1 \
    --save_dir logs/test \
    --visualization
```

视频命名格式：`ep{ep_i}_success_{True/False}_steps_{N}.mp4`

视频内容：将 4 路相机画面（`obs["rgb"]`）拼成 2x2 网格：

```python
# scripts/evaluate_openpi.py 第 230-232 行
frames_grid = [np.vstack([np.hstack(f[:2]), np.hstack(f[2:4])]) for f in frames]
mediapy.write_video(video_path, frames_grid, fps=10)
```

| 左上 (0,0) | 右上 (0,1) |
|---|---|
| 左下 (1,0) | 右下 (1,1) |

`f[:2]` = `[second_image, depth]`，`f[2:4]` = `[front_image, wrist_image]`，fps=10。

---

## 11. 常见问题与排错

### 11.1 评测卡死在 `load_env`

**症状**：日志停在 `load_env 开始` 之后无输出。

**根因**：通常是某个 `Condition.is_met()` 在 `record_initial_state` 未调用时错误返回 True，导致 `should_terminate_episode → reset → step → terminate` 死循环。

**解决**：参考 [`TROUBLESHOOTING.md` 第 3 节](TROUBLESHOOTING.md#3-liftcondition-在-record_initial_state-未调用时误判为已满足)。`LiftCondition.is_met()` 已修复（`VLABench/tasks/condition.py` 第 367-372 行）。

### 11.2 `TypeError: 'NoneType' object is not iterable`

**症状**：评测启动后立即报错。

**根因**：任务没有在 `VLABench/configs/__init__.py` 的 `name2config` 中注册，导致 `find_key_by_value(name2config, task_name)` 返回 `None`。

**解决**：在 `name2config` 中添加映射：

```python
"new_task_series": ["new_task"],
```

### 11.3 `ModuleNotFoundError: No module named 'openpi_client'`

**症状**：客户端启动时报错。

**解决**：
```bash
conda activate vlabench_2
pip install openpi-client
```

### 11.4 MuJoCo 段错误 / 渲染失败

**症状**：客户端报 `Segfault` 或 `AttributeError`。

**解决**：确保设置了 `MUJOCO_GL=egl`：
```bash
export MUJOCO_GL=egl
```

### 11.5 `jaxlib.xla_extension.XlaRuntimeError`

**症状**：服务器启动失败，BLAS 错误。

**根因**：GPU 显存不足或选错了 GPU。

**解决**：
```bash
nvidia-smi  # 检查 GPU 显存
export CUDA_VISIBLE_DEVICES=1  # 切换 GPU
```

### 11.6 网络问题（无法下载 checkpoint / tokenizer）

**症状**：服务器启动时下载失败。

**解决**：使用 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 11.7 评测超时（200 步还没成功）

**症状**：`success=False, steps=200`。

**根因**：策略模型对当前任务泛化能力不足。常见情况：
- 训练数据未涵盖任务中的物体（如 `lift_flask` 用 flask 训练过，pi0-fast-primitive-10task 不一定见过）
- 任务需要多步操作（pick → shake），模型难以完成

**建议**：
- 查看保存的视频确认具体失败位置
- 简化任务或增加训练数据
- 对多步任务，pi0-fast 表现通常有限

### 11.8 导入 `evaluate_openpi.py` 后任务数不对

**症状**：调试输出显示"已注册的任务数"少于预期。

**根因**：`autogen_tasks` 动态加载依赖于 `from VLABench.tasks import *` 已经执行过。检查导入顺序。

**解决**：确认 `evaluate_openpi.py` 中：
```python
from VLABench.robots import *       # 先 import VLABench.tasks 触发 register 初始化
from VLABench.tasks import *         # ← 这一步必须在前
# ... 然后才是动态加载 autogen_tasks
```

---

## 12. 快速命令参考

### 12.1 启动服务器

```bash
source /ssd/mkqin/workspace/VLABench/third_party/openpi/examples/vlabench/.venv/bin/activate
export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1
cd /ssd/mkqin/workspace/VLABench/third_party/openpi
python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pifast_ft_vlabench_primitive \
    --policy.dir=/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task
```

### 12.2 评测单个任务

```bash
source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2
export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
cd /ssd/mkqin/workspace/VLABench
python scripts/evaluate_openpi.py \
    --tasks lift_small_beaker \
    --n_episode 1 \
    --save_dir logs/lift_small_beaker \
    --visualization
```

### 12.3 评测多个任务

```bash
python scripts/evaluate_openpi.py \
    --tasks lift_beaker lift_small_beaker pick_tube_lift_tube \
    --n_episode 3 \
    --save_dir logs/multi_task \
    --visualization
```

### 12.4 后台启动服务器

```bash
nohup python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pifast_ft_vlabench_primitive \
    --policy.dir=/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task \
    > /tmp/openpi_server.log 2>&1 &

# 监控日志
tail -f /tmp/openpi_server.log

# 等待服务器就绪（看到 "server listening on 0.0.0.0:8000" 即可）
```

### 12.5 一键脚本

保存为 `run_openpi_eval.sh`：

```bash
#!/bin/bash
# 用法: ./run_openpi_eval.sh {server|eval} [task_name]

PORT=8000
GPU_ID=1
CHECKPOINT=/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task
SAVE_DIR=/ssd/mkqin/workspace/VLABench/logs/openpi_eval
TASK=${2:-lift_small_beaker}

start_server() {
    source /ssd/mkqin/workspace/VLABench/third_party/openpi/examples/vlabench/.venv/bin/activate
    export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
    export HF_ENDPOINT=https://hf-mirror.com
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    cd /ssd/mkqin/workspace/VLABench/third_party/openpi
    python scripts/serve_policy.py --port $PORT policy:checkpoint \
        --policy.config=pifast_ft_vlabench_primitive \
        --policy.dir=$CHECKPOINT
}

run_eval() {
    source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
    conda activate vlabench_2
    export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
    export HF_ENDPOINT=https://hf-mirror.com
    export MUJOCO_GL=egl
    cd /ssd/mkqin/workspace/VLABench
    python scripts/evaluate_openpi.py \
        --tasks "$TASK" \
        --n_episode 1 \
        --save_dir $SAVE_DIR \
        --visualization
}

case "$1" in
    server) start_server ;;
    eval) run_eval ;;
    *) echo "Usage: $0 {server|eval} [task_name]"; exit 1 ;;
esac
```

使用：
```bash
chmod +x run_openpi_eval.sh

# 终端 1: 启动服务器
./run_openpi_eval.sh server

# 终端 2: 评测（默认 lift_small_beaker）
./run_openpi_eval.sh eval
./run_openpi_eval.sh eval pick_tube_lift_tube
```

---

## 附：相关文件

| 文件 | 作用 |
|------|------|
| `scripts/evaluate_openpi.py` | 评测客户端入口（新） |
| `third_party/openpi/scripts/serve_policy.py` | 服务器入口 |
| `third_party/openpi/examples/vlabench/eval.py` | 旧版评测客户端（tyro + Evaluator，**已弃用**） |
| `VLABench/configs/__init__.py` | `name2config` 任务名映射 |
| `VLABench/configs/task_config.json` | 任务基础配置（含 `evaluation.max_episode_length`） |
| `VLABench/configs/constant.py` | 实体注册（`name2class_xml`） |
| `VLABench/envs/dm_env.py` | MuJoCo 仿真环境 (`LM4ManipDMEnv`) |
| `VLABench/tasks/dm_task.py` | 任务基类 (`LM4ManipBaseTask`) |
| `VLABench/tasks/hierarchical_tasks/primitive/` | 手写 primitive 任务 |
| `VLABench/tasks/autogen_tasks/` | 自动生成的 primitive 任务 |
| `VLABench/tasks/autogen_tasks/base.py` | autogen 任务的 `PrimitiveTask` 基类 |
| `VLABench/tasks/condition.py` | 成功条件系统（`LiftCondition` 等） |
| `VLABench/utils/register.py` | 任务/实体/条件注册系统 |
| `docs/TROUBLESHOOTING.md` | 错误排错文档（`LiftCondition` bug 在第 3 节） |

---

**最后更新**: 2026-06-03
**维护者**: VLABench Team
