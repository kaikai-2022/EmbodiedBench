# OpenPI 模型评测技术文档

## 1. 系统架构：Client-Server 模式

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Terminal 1: OpenPI Policy Server (OpenPI虚拟环境 .venv)                  │
│                                                                         │
│  serve_policy.py                                                        │
│      ├── 加载 pi0-fast 模型权重 (JAX/Flax)                              │
│      ├── 加载 PaliGemma tokenizer                                        │
│      └── 启动 WebSocket 服务器, 监听 8000 端口                          │
│                                                                         │
│  接收: policy_input = {image, wrist_image, state, prompt}               │
│  返回: {"actions": [...]}                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │ WebSocket (msgpack序列化)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Terminal 2: VLABench Evaluator Client (vlabench_2 conda环境)           │
│                                                                         │
│  eval.py                                                                │
│      ├── Pi0: WebSocket客户端, 封装predict()调用                         │
│      ├── Evaluator: 评测循环, 管理环境, 计算指标                        │
│      └── VLABench环境 (MuJoCo): 仿真执行, 渲染图像, 条件判断            │
└─────────────────────────────────────────────────────────────────────────┘
```

**为什么分两个终端/两个环境？**
- **服务器端**需要 OpenPI 的依赖（JAX、Flax等），需要在 GPU 上运行
- **客户端**需要 VLABench 的依赖（MuJoCo、dm_control 等），管理仿真环境
- 两者通过 **WebSocket** 解耦，通过 msgpack 序列化传输数据

---

## 2. 文件清单与角色

| 文件路径 | 角色 | 说明 |
|---------|------|------|
| `third_party/openpi/scripts/serve_policy.py` | 服务器入口 | 加载模型, 启动WebSocket服务 |
| `third_party/openpi/src/openpi/policies/policy.py` | 模型推理核心 | `_sample_actions()` 方法 |
| `third_party/openpi/src/openpi/serving/websocket_policy_server.py` | WebSocket服务 | 接收请求, 调用policy.infer() |
| `third_party/openpi/src/openpi/models/model.py` | 模型网络结构 | 定义pi0-fast的JAX网络 |
| `third_party/openpi/examples/vlabench/eval.py` | 评测客户端入口 | Pi0策略类, Evaluator, 评测循环 |
| `VLABench/evaluation/evaluator/base.py` | 评测引擎 | `Evaluator.evaluate()`, `evaluate_single_episode()` |
| `VLABench/envs/dm_env.py` | MuJoCo仿真环境 | `load_env()`, `LM4ManipDMEnv` 类 |
| `VLABench/tasks/dm_task.py` | 任务基类 | `LM4ManipBaseTask`, 实体加载, 条件判断 |
| `VLABench/tasks/hierarchical_tasks/primitive/base.py` | 原子任务基类 | `PrimitiveTask`, `get_intention_score()` |
| `VLABench/tasks/hierarchical_tasks/primitive/xxx_series.py` | 具体任务实现 | `get_instruction()`, `get_expert_skill_sequence()`, 成功条件 |
| `VLABench/configs/task_config.json` | 任务基础配置 | 场景, 物体清单, 容器配置 |
| `VLABench/configs/constant.py` | 全局实体注册 | `name2class_xml` 字典 |
| `VLABench/configs/__init__.py` | 任务名映射 | `name2config` 映射 series名→任务名 |
| `VLABench/utils/register.py` | 注册系统 | 全局注册表, `@register.add_task()`, `@register.add_config_manager()` |
| `VLABench/tasks/condition.py` | 成功条件系统 | `Condition`, `contain`, `distance` 等 |
| `VLABench/utils/skill_lib.py` | 脚本技能库 | `SkillLib.pick()`, `SkillLib.lift()` 等 |

---

## 3. 完整调用链路

```
python eval.py --args.tasks "shake_tube" --args.n_episode 1
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ eval.py:main()                                                         │
│                                                                        │
│ 1. tyro.cli(main) 解析命令行参数                                         │
│ 2. episode_configs = None (没有用eval_track)                             │
│ 3. tasks = ["shake_tube"]                                              │
│ 4. client = WebsocketClientPolicy("localhost", 8000)                   │
│ 5. policy = Pi0(client, replan_steps=5)                                │
│ 6. evaluator = Evaluator(tasks, n_episodes=1, ...)                      │
│ 7. evaluator.evaluate(policy)                                           │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Evaluator.evaluate(policy)                                              │
│                                                                        │
│ for task in ["shake_tube"]:                                             │
│   for i in [0]:  (n_episodes=1)                                        │
│     agent.reset()           # Pi0.action_plan.clear()                 │
│     evaluate_single_episode(agent, task, i, None, seed=42)              │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Evaluator.evaluate_single_episode()                                      │
│                                                                        │
│ 1. env = load_env("shake_tube", random_init=True)  ←─── 关键步骤     │
│ 2. env.reset()                                                         │
│ 3. while step < max_episode_length(200):                               │
│      ├── obs = env.get_observation()        # 获取RGB+状态            │
│      ├── obs["instruction"] = env.task.get_instruction()  # 语言指令  │
│      ├── pos, euler, gripper = agent.predict(obs)  # ←── 关键调用   │
│      ├── action = [...pos, ...euler, ...gripper]                      │
│      ├── timestep = env.step(action)           # MuJoCo执行           │
│      └── if timestep.last(): success=True; break                      │
│ 4. intention_score = env.get_intention_score()                          │
│ 5. progress_score = env.get_task_progress()                            │
│ 6. return {"task", "success", "consumed_step", "intention_score", ...}│
└───────────────────────────────────────────────────────────────────────┘
        │
        ├──► Pi0.predict(obs)                                          │
        │                                                           │
        │   if action_plan为空:                                          │
        │     # 准备发送给OpenPI服务器的数据                            │
        │     policy_input = {                                          │
        │       "observation/image": front_rgb,                          │
        │       "observation/second_image": second_rgb,                 │
        │       "observation/wrist_image": wrist_rgb,                   │
        │       "observation/state": [pos, euler, gripper],            │
        │       "prompt": "Pick up the CuCl2 tube..."                │
        │     }                                                         │
        │     actions = model.infer(policy_input)["actions"]  # WebSocket│
        │     action_plan.extend(actions[:5])                            │
        │   action = action_plan.popleft()                             │
        │   return target_pos, target_euler, gripper_state             │
        │                                                           │
        └──► load_env("shake_tube", ...)  ←─── 环境加载             │
                                                                    │
            1. name2config["shake_tube"] → "shake_tube_series"         │
            2. TASK_CONFIG["shake_tube_series"] → {场景, 物体, 容器}    │
            3. robot = load_robot("franka")(...)                       │
            4. task = ShakeTubeTask(task, robot, ...)  ←─── 任务实例化 │
            5. env = LM4ManipDMEnv(task, ...)                          │
            6. env.reset() → task.build_from_config()                 │
```

---

## 4. 任务注册机制

### 4.1 注册装饰器

所有任务通过全局注册系统注册：

```python
# primitive/xxx_series.py
@register.add_config_manager("task_name")
class XXXConfigManager(BenchTaskConfigManager):
    ...

@register.add_task("task_name")
class XXXTask(PrimitiveTask):
    ...
```

### 4.2 任务名解析

```
eval.py 中 --args.tasks "shake_tube"
        │
        ▼
name2config["shake_tube"] → "shake_tube_series"   (configs/__init__.py)
        │
        ▼
TASK_CONFIG["shake_tube_series"] → {场景, 物体, 容器配置}  (configs/task_config.json)
        │
        ▼
register.load_task("shake_tube") → ShakeTubeTask 类  (utils/register.py)
```

### 4.3 实体注册

```python
# configs/constant.py
name2class_xml = {
    "tube": [ChemistryTube, "obj/meshes/tube/tube/tube.xml"],
    "chemistry_tube_stand": [TubeStand, "obj/meshes/tube/tube_container/tube_stand.xml"],
    ...
}
```

---

## 5. 环境加载详细流程

```
load_env("shake_tube", random_init=True)
        │
        ▼
1. robot = register.load_robot("franka")(robot_config)
        │
        ▼
2. task = register.load_task("shake_tube")(task, robot, ...)
   → ShakeTubeTask.__init__(task, robot, ...)
        │
        ▼
3. ShakeTubeTask.build_from_config()
   ├── BenchTaskConfigManager.__init__()    # 初始化配置管理器
   │   ├── self.config = {task: {...}}
   │   ├── self.seen_object = ["CuCl2", "CuSO4", "FeCl3", "KMnO4"]
   │   ├── self.unseen_object = ["I2", "K2CrO4"]
   │   └── self.target_entity = random.choice(seen_object)
   ├── config_manager.load_init_containers()  # 加载试管架
   ├── config_manager.load_objects()        # 加载试管作为子实体
   │   # 注意: 试管不作为独立entities, 而是通过tube_stand.subentities加载
   │   # 会设置self.target_tube_hole记录目标试管位置
   ├── config_manager.get_instruction()       # "Pick up the CuCl2 tube..."
   └── config_manager.get_condition_config() # contain(container, entities)
        │
        ▼
4. task.build_from_config(eval=True)
   ├── TaskConfigManager.build()  # 构建所有entities
   ├── for entity in task.entities: detach/attach到arena
   └── ShakeTubeTask自定义: 固定试管架位置
        │
        ▼
5. env = LM4ManipDMEnv(task, time_limit, reset_wait_step)
6. env.reset() → task.setup_episode()
   ├── 启用重力/流体仿真
   └── 返回初始timestep
```

---

## 6. OpenPI 模型推理流程

```
Pi0.predict(obs) 被 Evaluator 调用
        │
        ▼
action_plan为空?
  │
  ├── Yes: 构造 policy_input
  │       ├── "observation/image": 正面相机RGB (480x640)
  │       ├── "observation/second_image": 第二相机RGB
  │       ├── "observation/wrist_image": 手腕相机RGB
  │       ├── "observation/state": [pos, euler, gripper_state]
  │       └── "prompt": "Pick up the CuCl2 tube from the stand..."
  │       │
  │       ▼
  │   model.infer(policy_input)  ←─── WebSocket发送
  │       │
  │       ├── client._ws.send(msgpack.pack(obs))
  │       ├── server接收 → policy.infer(obs)
  │       │    ├── _model.Observation.from_dict(inputs)
  │       │    ├── _sample_actions(sample_rng, observation, **kwargs)
  │       │    │    └── 返回actions数组
  │       │    └── return {"actions": actions}
  │       ├── msgpack.unpack(server_response)
  │       └── return {"actions": [...]}
  │
  └── No: 直接从action_plan取动作

返回: target_pos, target_euler, gripper_state
      (Evaluator再转成关节qpos执行)
```

---

## 7. 评测指标计算

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| `success_rate` | `env.check_success()` | 仿真器判断所有`Condition`是否满足 |
| `intention_score` | `env.get_intention_score()` | 末端执行器是否接近目标实体（pick类任务有意义） |
| `progress_score` | `env.get_task_progress()` | 技能序列执行进度（条件满足比例） |

**shake_tube 任务特殊处理：**
- 试管作为子实体加载，不在`self.entities`中
- 因此 `reset_intention_distance()` 不会初始化 `intention_distance["CuCl2"]`
- 查表时会 `KeyError`
- **解决：重写 `get_intention_score()` 返回固定值**

```python
# shake_tube_series.py
def get_intention_score(self, physics, threshold=0.2, discrete=True):
    return 0  # 这个任务不需要intention分数, 只看success_rate
```

---

## 8. 成功条件系统

成功条件在 `get_condition_config()` 中定义：

```python
# shake_tube_series.py
def get_condition_config(self, target_entity, init_container, ...):
    conditions_config = dict(
        contain=dict(
            container=init_container,  # chemistry_tube_stand
            entities=[target_entity]      # CuCl2
        )
    )
```

条件判断在 `Condition.met(physics)` 中：

```python
# dm_task.py
conditions.met(physics) → (all_met, met_list)
    └── for condition in conditions:
            condition.met(physics) → bool
```

对于 `contain` 条件：
- 检查实体是否在容器的包围盒内
- 需要实体的`AABBox`和容器的`AABBox`

---

## 9. eval.py 中的 Pi0 策略类

```python
class Pi0(Policy):
    def __init__(self, client, replan_steps=5):
        self.model = client               # WebsocketClientPolicy
        self.replan_steps = replan_steps # 每5步重新规划
        self.action_plan = deque(maxlen=5) # 动作缓冲

    def predict(self, obs, **kwargs):
        # obs来自Evaluator, 包含:
        #   obs["rgb"] = [second_img, depth, front_img, wrist_img]
        #   obs["ee_state"] = [pos(3), quat(4), gripper(1)]
        #   obs["instruction"] = "Pick up the CuCl2 tube..."
        #   obs["last_action"] = [pos, euler]

        if len(self.action_plan) == 0:  # 缓冲空, 重新推理
            policy_input = {
                "observation/image": front_rgb,
                "observation/second_image": second_rgb,
                "observation/wrist_image": wrist_rgb,
                "observation/state": [pos-offset, euler, gripper],
                "prompt": obs["instruction"]
            }
            actions = self.model.infer(policy_input)["actions"]
            self.action_plan.extend(actions[:self.replan_steps])

        action = self.action_plan.popleft()
        return target_pos, target_euler, gripper_state
```

---

## 10. VLABench 评测的其他 Policy 实现

| Policy | 文件 | 用途 |
|--------|------|------|
| `Pi0` | `third_party/openpi/examples/vlabench/eval.py` | **OpenPI主评测** |
| `OpenPiPolicy` | `VLABench/evaluation/model/policy/openpi.py` | 旧版策略（V1） |
| `OpenVLA` | `VLABench/evaluation/model/policy/openvla.py` | OpenVLA模型 |
| `Gr00tPolicy` | `VLABench/evaluation/model/policy/gr00t.py` | Gr00t模型 |
| `RandomPolicy` | `VLABench/evaluation/model/policy/base.py` | 随机基线 |

---

## 11. 新增自定义任务的检查清单

新增一个自定义任务（如`shake_tube`）需要检查：

### 11.1 必须完成

| # | 检查项 | 文件位置 | 状态 |
|---|--------|---------|------|
| 1 | 创建 `XxxTask` 类, 用 `@register.add_task("xxx")` 注册 | `primitive/xxx_series.py` | ✅ shake_tube已完成 |
| 2 | 创建 `XxxConfigManager` 类, 用 `@register.add_config_manager("xxx")` 注册 | `primitive/xxx_series.py` | ✅ |
| 3 | 在 `primitive/__init__.py` 添加导入 | `from .xxx_series import *` | ✅ |
| 4 | 在 `configs/__init__.py` 添加 series→task 映射 | `name2config["xxx_series"] = ["xxx"]` | ✅ |
| 5 | 在 `configs/task_config.json` 添加任务配置 | 场景、物体、容器 | ✅ |
| 6 | 实现 `get_instruction()` 返回语言指令 | `ConfigManager` 方法 | ✅ |
| 7 | 实现 `get_condition_config()` 定义成功条件 | `ConfigManager` 方法 | ✅ |
| 8 | 实现 `get_expert_skill_sequence()` 定义参考技能序列 | `Task` 方法 | ✅ |
| 9 | 在 `configs/constant.py` 注册所有实体 | `name2class_xml` | ⚠️ 动态加载的实体可不注册 |

### 11.2 常见问题

**Q: `intention_distance` KeyError**
- 原因：目标实体作为子实体加载，不在`self.entities`中
- 解决：重写`get_intention_score()`返回固定值

**Q: 评测指标全是NaN**
- 原因：同上，KeyError导致episode信息为空
- 解决：重写`get_intention_score()`

**Q: MuJoCo渲染段错误**
- 原因：headless环境下EGL渲染问题
- 解决：`export MUJOCO_GL=egl`

---

## 12. 快速命令参考

### 启动服务器（终端1）
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

### 运行评测（终端2）
```bash
source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2
export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
cd /ssd/mkqin/workspace/VLABench/third_party/openpi
python examples/vlabench/eval.py \
    --args.host localhost \
    --args.port 8000 \
    --args.tasks "shake_tube" \
    --args.n_episode 10 \
    --args.save_dir /ssd/mkqin/workspace/VLABench/logs/openpi_eval
```
