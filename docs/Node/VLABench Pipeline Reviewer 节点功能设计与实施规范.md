# VLABench Pipeline: `Reviewer` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Reviewer` 节点是整个 VLABench Pipeline 的"任务质量审核器（Task Quality Reviewer）"，位于 `Simulation` 节点之后、`VLM Data` 节点之前。

它承接 `Simulation` 节点生成的视频和 `Condition Planner` 输出的条件，利用 LLM 的视觉理解能力对每个执行步骤进行逐一审核，判断是否满足预期的成功条件。

**核心职责：**

1. **视频帧提取**：根据 `Simulation` 记录的时间戳从视频中提取每个 step 的关键帧
2. **批量审核**：将所有 step 的截图和 condition 整合到单一 prompt 中，一次性发送给 LLM
3. **结果汇总**：解析 LLM 返回的 JSON 结果，输出每个 step 的通过/失败状态和原因
4. **决策输出**：生成 `review_results` 和 `review_passed` 供后续节点使用

---

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `state["step_timestamps"]` | List[Dict] | Simulation | 每个 step 和原子操作的时间戳 |
| `state["simulation_video_path"]` | String/None | Simulation | 生成的视频路径 |
| `state["condition_plan"]` | List[Dict] | Condition Planner | 每个 step 的成功条件和 reasoning |
| `state["task_analysis"]` | Dict | Analyzer | 包含 `instruction_en` 用于 LLM 理解任务 |
| `state["asset_status"]` | Dict | Asset Manager | 场景资产信息（uid → class_name） |

### 2.2 输出 (Output to State)

| 字段 | 类型 | 说明 |
|------|------|------|
| `state["review_results"]` | List[Dict] | 每个 step 的审核结果：`[{"step_id": int, "passed": bool, "reason": str}]` |
| `state["review_passed"]` | Bool | 全部通过为 `True`，否则为 `False` |
| `state["current_stage"]` | String | 始终为 `"vlm_data"` |

---

## 3. 执行流程 (Execution Flow)

```
1. 检查 langchain_anthropic 是否可用（不可用时默认通过）
2. 获取必要信息：step_timestamps, video_path, condition_plan, instruction, asset_status
3. 调用 extract_frames_from_timestamps() 从视频中提取帧
   - 对每个原子操作，提取 (start + end) / 2 时刻的帧
   - 对每个大 step，提取 step_end 时刻的帧
4. 调用 build_reviewer_prompt() 构建包含所有 step 的大 prompt
   - 将 condition_plan 中的 reasoning 字段作为 Condition 描述
5. 调用 LLM（支持 vision）获取审核结果
   - 有图片时使用多模态调用
   - 无图片时使用纯文本模式
6. 解析 LLM 返回的 JSON：{"results": [...], "all_passed": bool}
7. 输出 review_results 和 review_passed 到 state
8. 进入 vlm_data 节点
```

---

## 4. 时间戳数据结构

### 4.1 Simulation 输出的 step_timestamps

```python
{
    "step_timestamps": [
        {
            "step_id": 0,
            "atomic_timestamps": [
                {"atomic_idx": 0, "start": 0.0, "end": 1.5},
                {"atomic_idx": 1, "start": 1.5, "end": 3.2}
            ],
            "step_end": 3.2
        },
        {
            "step_id": 1,
            "atomic_timestamps": [
                {"atomic_idx": 2, "start": 3.2, "end": 4.5},
                {"atomic_idx": 3, "start": 4.5, "end": 6.0}
            ],
            "step_end": 6.0
        }
    ]
}
```

### 4.2 时间戳记录时机

- **原子操作开始**：记录 `atomic_start = 当前时间 - video_start_time`
- **原子操作结束**：记录 `atomic_end = 当前时间 - video_start_time`
- **大 step 结束**：当 step 的最后一个技能执行完成时，记录 `step_end`

---

## 5. 视频帧提取策略

### 5.1 extract_frames_from_timestamps 函数

```python
def extract_frames_from_timestamps(video_path: str, timestamps: List[Dict]) -> Dict[int, List[str]]:
    """
    根据时间戳从视频中提取帧。

    返回: {step_id: [frame_paths]}，每个 step 的帧路径列表
    """
```

### 5.2 帧提取逻辑

1. 打开视频获取 fps 和总帧数
2. 对每个 step：
   - 对每个原子操作，计算中点时间 `(start + end) / 2`，转换为帧索引
   - 提取该帧并保存到临时目录
   - 对大 step 结束后，提取 step_end 时刻的帧
3. 返回 `{step_id: [frame_paths]}`

### 5.3 帧提取时机

| 帧类型 | 提取时间 | 说明 |
|--------|----------|------|
| 原子操作中点帧 | `(atomic_start + atomic_end) / 2` | 展示技能执行到一半的状态 |
| 大 step 完成后帧 | `step_end` | 展示整个 step 完成后的状态 |

---

## 6. LLM Prompt 设计

### 6.1 Prompt 结构

```markdown
## 任务
你是一个机器人任务质量审核员。你的职责是根据仿真视频截图和成功条件，判断每个执行步骤是否正确完成。

### 任务指令
{instruction}

### 场景资产
{asset_status}

### 执行步骤列表
[
  {
    "step_id": 0,
    "screenshots": [frame1.jpg, frame2.jpg, ...],
    "Condition": "The step is to pick rag_0, so the success condition is that rag_0 is grasped by robot"
  },
  {
    "step_id": 1,
    "screenshots": [frame3.jpg, frame4.jpg, ...],
    "Condition": "The step is to lift rag_0, so the success condition is that rag_0 is above the target height threshold"
  },
  ...
]

### 审核要求
1. 逐一检查每个 step 的截图，判断 condition 是否满足
2. 如果某个 step 失败，说明图片中显示的具体问题
3. 如果某个 step 通过，给出简短的理由

### 输出格式（严格 JSON，无其他文本）
{
  "results": [
    {"step_id": 0, "passed": true, "reason": "..."},
    {"step_id": 1, "passed": false, "reason": "..."}
  ],
  "all_passed": false
}
```

### 6.2 Condition 字段来源

直接使用 `Condition Planner` 输出的 `reasoning` 字段，这是 LLM 生成的自然语言解释，说明该 step 的成功条件是什么。

### 6.3 LLM 调用方式

- **有图片时**：使用 vision model，将截图作为 base64 编码的图像块发送
- **无图片时**：使用纯文本模式，prompt 中只包含步骤描述

---

## 7. 输出示例

### 7.1 成功场景

```python
{
    "review_results": [
        {"step_id": 0, "passed": True, "reason": "抓取动作正确完成，rag_0 位于机器人抓夹中"},
        {"step_id": 1, "passed": True, "reason": "放置动作正确完成，rag_0 已移动到 rack_0 顶部"}
    ],
    "review_passed": True,
    "current_stage": "vlm_data"
}
```

### 7.2 失败场景

```python
{
    "review_results": [
        {"step_id": 0, "passed": True, "reason": "抓取动作正确完成"},
        {"step_id": 1, "passed": False, "reason": "图片显示 rag_0 仍在台面上，未被移动到目标位置"}
    ],
    "review_passed": False,
    "current_stage": "vlm_data"
}
```

---

## 8. 错误处理与降级策略

| 场景 | 处理方式 |
|------|----------|
| langchain_anthropic 未安装 | 输出空的 review_results，review_passed 默认为 True |
| 视频文件不存在 | 输出空的 review_results，review_passed 默认为 True |
| 无法提取帧 | 输出空的 review_results，review_passed 默认为 True |
| LLM JSON 解析失败 | 重试最多 2 次，失败后默认通过 |
| LLM 调用异常 | 重试最多 2 次，失败后默认通过 |

---

## 9. 与 Simulation 节点的配合

### 9.1 时间戳记录

`Simulation` 节点在技能执行过程中记录时间戳：

```python
video_start_time = time.time()

for skill_idx, skill in enumerate(skill_seq):
    atomic_start = time.time() - video_start_time
    skill(env)
    atomic_end = time.time() - video_start_time
    current_atomic_timestamps.append({
        "atomic_idx": skill_idx,
        "start": atomic_start,
        "end": atomic_end
    })

    # 当 step 完成时
    if skill_idx == step_skill_ends[step_idx] - 1:
        step_timestamps.append({
            "step_id": step_idx,
            "atomic_timestamps": current_atomic_timestamps,
            "step_end": time.time() - video_start_time
        })
        current_atomic_timestamps = []
```

### 9.2 输出到 State

`Simulation` 节点输出：
```python
{
    "step_timestamps": [...],
    "current_stage": "reviewer"
}
```

---

## 10. 重要注意事项

### 10.1 帧提取的边界检查

提取帧时需进行边界检查，确保帧索引在有效范围内：

```python
frame_idx = max(0, min(frame_idx, total_frames - 1))
```

### 10.2 临时文件清理

提取的帧保存在临时目录（`tempfile.mkdtemp`），LLM 调用完成后自动由操作系统清理。

### 10.3 图片数量限制

如果 step 数量很多且每个 step 的原子操作数量也多，图片数量会成倍增加。建议：
- 限制每个 step 最多 5 个原子操作（对应 6 张图：5 个中点帧 + 1 个结束帧）
- 如果超过限制，可以采样或跳过部分帧

### 10.4 回退功能（暂未实现）

当前版本的 Reviewer 节点仅输出审核结果，不包含自动回退机制。后续可扩展：
- 当 `review_passed = False` 时，根据失败的 step 决定回退到哪个节点
- 利用 LLM 的分析能力判断失败原因（skill_planner / code_generator / normalizer 等）