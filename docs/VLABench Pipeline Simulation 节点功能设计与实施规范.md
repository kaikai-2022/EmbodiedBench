# VLABench Pipeline: `Simulation` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Simulation` 节点是整个 VLABench Pipeline 的"物理仿真执行器（Physics Simulation Executor）"，也是 Pipeline 的最终验证关卡。

它承接 `Registration` 节点注册好的任务，在 MuJoCo 物理引擎中加载场景、执行专家技能序列、收集观测数据，并判断任务是否成功完成。

**核心职责：**

1. **动态加载 series 文件**：在标准 primitive 包导入之外，额外 `importlib` 加载 `<task_name>_series.py`，确保 `@register` 装饰器被触发。
2. **执行技能序列**：逐个调用 `get_expert_skill_sequence()` 返回的 `partial` 技能，收集观测和轨迹。
3. **超时保护**：单个技能 300s，整体仿真 600s，用 `SIGALRM` 强制中断。
4. **数据持久化**：仿真成功时保存 HDF5 训练数据，无论成功/失败都保存视频。
5. **错误反馈**：失败时将详细错误信息写入 `error_feedback`，供上游节点重试参考。

---

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `state["task_analysis"]` | Dict | Analyzer | 包含 `task_name`，用于加载环境 |

### 2.2 输出 (Output to State)

| 字段 | 类型 | 说明 |
|------|------|------|
| `state["simulation_success"]` | Bool | 任务是否成功完成 |
| `state["simulation_video_path"]` | String/None | 生成的视频路径 |
| `state["simulation_hdf5_path"]` | String/None | HDF5 数据路径（仅成功时） |
| `state["episode_config"]` | Dict/None | 本次 episode 的确定性配置 |
| `state["executed_skill_sequence"]` | List | 实际执行的技能序列（`operation_sequence` 格式） |
| `state["error_feedback"]` | String/None | 失败原因（成功时为 None） |
| `state["current_stage"]` | String | 成功时为 `"vlm_data"`，失败时为 `"simulation"` |

---

## 3. 执行流程 (Execution Flow)

```
1. 设置 VLABENCH_ROOT 环境变量
2. import VLABench.robots / tasks（触发 @register）
3. 动态加载 <task_name>_series.py（触发 task 的 @register）
4. load_env(task_name, robot="franka") 构建 MuJoCo 场景
5. 获取 get_expert_skill_sequence() 返回的技能列表
6. 循环执行每个技能，收集 observations 和 waypoints
7. 技能返回 (obs, waypoints, stage_success, task_success)
   - stage_success=False → 中断，仿真失败
   - task_success=True → 任务完成，跳出循环
8. 执行完后检查 env.task.conditions.is_met()（兜底）
9. 保存视频（无论成功/失败）
10. 成功时：保存 HDF5 数据
11. 关闭环境 env.close()
```

---

## 4. series 文件动态加载机制

`simulation_node` 在标准 `importlib.import_module("VLABench.tasks.hierarchical_tasks.primitive")` 之外，额外动态加载 series 文件：

```python
series_path = $VLABENCH_ROOT/tasks/hierarchical_tasks/primitive/<task_name>_series.py
if exists(series_path):
    importlib.util.spec_from_file_location(...).loader.exec_module(mod)
```

**原因**：`primitive` 包的 `__init__.py` 不会自动 import 动态生成的 series 文件，必须显式加载才能触发 `@register` 装饰器。

**扩展**：如需支持其他目录（如 `composite`）的动态任务，可仿照此方式添加对应路径的动态加载逻辑。

---

## 5. 技能执行接口

每个技能是一个 `functools.partial` 对象，调用签名为：

```python
obs, waypoints, stage_success, task_success = skill(env)
```

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `obs` | List[Dict] | 每步的观测帧（含 RGB、point cloud 等） |
| `waypoints` | List[np.ndarray] | 末端执行器轨迹点 `[pos(3) + euler(3) + gripper(2)]` |
| `stage_success` | Bool | 当前技能是否成功完成 |
| `task_success` | Bool | 整体任务是否完成（部分技能可提前判定） |

---

## 6. 超时保护机制

使用 `signal.SIGALRM` 实现两级超时：

```python
SKILL_TIMEOUT = 300   # 单技能超时（秒）
SIMULATION_TIMEOUT = 600  # 整体超时（秒）
```

每个技能开始前重置单技能超时计时器；整体超时在进入技能循环前设置。超时触发 `SkillTimeoutError`，节点捕获后返回失败。

---

## 7. 数据保存

### 7.1 视频（始终保存）

```
$PROJECT_ROOT/dataset/training_data/<task_name>/demo_0_success_<True/False>.mp4
```

每帧由 4 个摄像头视角拼接（2×2 网格），帧率 10fps。

### 7.2 HDF5 训练数据（仅成功时）

```
$PROJECT_ROOT/dataset/training_data/<task_name>/data_0.hdf5
```

包含：`trajectory`（机器人坐标系的轨迹点）、`entities`、`target_entity`、`episode_config`、`instruction`，以及 `process_observations` 处理后的多模态观测数据。

---

## 8. 错误反馈格式

仿真失败时 `error_feedback` 包含：

- **技能执行失败**：`"仿真执行完毕但任务未成功完成。执行的技能: [...] 请检查 get_expert_skill_sequence ..."`
- **异常崩溃**：完整的 Python traceback 字符串

此信息会被传递给上游（如 Skill Planner 重试）以辅助 LLM 分析失败原因。

---

## 9. 重要注意事项

### 9.1 VLABENCH_ROOT 路径

必须设置为 VLABench **子目录**（含 `tasks/`、`assets/` 的那一层）：

```
VLABENCH_ROOT = /path/to/VLABench/VLABench  # ✓ 正确
VLABENCH_ROOT = /path/to/VLABench            # ✗ 错误（少一层）
```

### 9.2 waypoints 坐标系转换

HDF5 中保存的 `trajectory` 是相对机器人基座的坐标（减去 `robot_position`），而非世界坐标。公式：

```python
robot_frame_wp = waypoint - np.concatenate([robot_position, np.zeros(5)])
```

### 9.3 条件兜底检查

即使所有技能执行完后 `task_success` 仍为 False，节点会额外调用 `env.task.conditions.is_met(physics)` 做一次兜底判断，允许某些任务通过物理条件（而非代码标志）来判定成功。
