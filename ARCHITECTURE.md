# VLAbench_LZR 项目架构说明

本仓库是 VLABench（kaikai-2022 的 fork）的 `dev_LZR` 分支，定位是 **"科学实验操作基准（LabBench / SciVLABench）"**：把原版日常场景（厨房/客厅/打麻将）的机器人 benchmark，改造成**实验室场景下操作烧杯/锥形瓶/移液管/温度计等实验器具**的 VLA/VLM benchmark。

> 这条线索可以从 git log 看出来：`拧瓶盖任务创建成功`、`修改实验室场景，深蓝色桌面`、`场景更换完成`。

## 目录

- [顶层结构](#顶层结构)
- [一、核心包 `VLABench/`](#一核心包-vlabench)
- [二、脚本与运行入口](#二脚本与运行入口)
- [三、与原版 VLABench 的差异](#三与原版-vlabench-的差异)
- [四、典型数据流](#四典型数据流)
- [五、环境与运行要点](#五环境与运行要点)

---

## 顶层结构

```
VLAbench_LZR/
├── VLABench/          # 核心 Python 包（仿真环境 + 任务 + 评测）
├── scripts/           # 命令行入口：数据生成、格式转换、评测
├── sh/                # 批量运行脚本（多卡数据生成 / 评测）
├── docker/            # 官方推荐运行环境
├── third_party/openpi # 子模块：Physical Intelligence 的 openpi（π0 / π0-FAST 策略）
├── src/rrt-algorithms # 子模块：RRT 运动规划
├── server/            # 轻量 Flask 服务（app.py + index.html）
├── docs/              # 中文设计文档、训练报告、LaTeX 论文稿、tutorials
├── requirements.txt / setup.py / README.md / run_openvla_eval.sh
```

其中 `third_party/openpi` 和 `src/rrt-algorithms` 是 git 子模块（见 `.gitmodules`）。

---

## 一、核心包 `VLABench/`

核心包分为七个子模块。

### 1. `envs/` — 仿真环境封装（很薄）

- `dm_env.py` → `LM4ManipDMEnv`：把 dm_control 的 `Physics` 包成 RL 风格 env，提供 `reset()` / `step()` / `render(camera_id=...)` / `get_observation()`。
- `__init__.py` → `load_env(task, robot, ...)`：按任务名查 config → 实例化 robot + task → 组装成 env。这是 tutorial 里那一行的入口。

### 2. `robots/` — 机器人本体

按形态分三类，都继承 `base.py`：

- `single_arm/`：**franka**（默认）、widowx、xarm
- `dual_arm/`：双臂
- `humanoid/`：h1（人形）

### 3. `tasks/` — 任务定义（本分支扩张最猛的地方）

```
tasks/
├── dm_task.py / condition.py / config_manager.py   # 任务基类与条件 / 配置管理
├── components/   # 任务可复用积木：entity / container / scene
├── autogen_tasks/   # 自动生成的"原语任务"：pick_xxx / place_xxx / pour_xxx / lift_xxx
│                    # 全是化学器具：conical_flask、florence_flask、pipette、
│                    # thermometer、tube、beaker、mechanical_pipette、funnel ...
├── hierarchical_tasks/
│   ├── primitive/   # 原语级（add_condiment、insert_flower、heat_chemistry_tube ...）
│   └── composite/   # 组合级（cook_dishes、make_juice、heat_food、take_chemistry_experiment、
│                    #           rearrange_chemistry_tube、play_mahjong / poker / snooker ...）
└── autogen_tasks/base.py  # 自动生成任务的基类（*_series.py 都是它的子类实例）
```

**原语任务（primitive）= 单步抓取/放置/倾倒；组合任务（composite）= 多步骤长程任务。** 化学实验任务（`take_chemistry_experiment`、`rearrange_chemistry_tube`、`heat_chemistry_tube`）是本分支相比上游的核心增量。

### 4. `configs/` — 全部配置（YAML / JSON 驱动）

```
configs/
├── task_related/   # experiment.json / math.json / painting.json / recipe.json
│                    #   （实验 / 数学 / 绘画 / 食谱 这几类任务的知识库）
├── camera/  model/  prompt/  evaluation/   # 相机、模型、提示词、评测配置
└── （任务名 → config 的映射由 name2config 维护）
```

### 5. `evaluation/` — 评测框架（本分支新增的重头戏）

统一接口评测两类模型：

```
evaluation/
├── evaluator/      # base.py + vlm.py：跑一轮 episode、收集 reward / 成功判定
├── metric.py       # 成功率等指标
└── model/
    ├── policy/     # 策略模型适配器：base / act / gr00t / openpi / openvla / rdt
    └── vlm/        # VLM 适配器：base / gpt4 / claude / gemini /  / qwen_vl /
                    #              intern_vl / llava / minicpm
```

**关键设计**：`model/policy/base.py` 和 `model/vlm/base.py` 定义统一抽象，加新模型只要写一个适配器。policy 适配的 ACT / GR00T / OpenPI / OpenVLA / RDT 都是当前主流 VLA / imitation 策略；`docs/` 下有 ACT 训练 / 部署的成功报告。

### 6. `pipeline/` — 自动化任务生成流水线（本分支原创，最复杂）

把"用自然语言描述 → 自动生成可运行任务"做成了一条 LLM 驱动的流水线，每个 node 是一个阶段：

```
pipeline/
├── agent.py / config.py / state.py          # 流水线编排、全局配置、状态传递
├── nodes/
│   ├── normalizer.py          # 归一化用户输入
│   ├── analyzer.py            # 分析任务需求
│   ├── asset_manager.py       # 管理 / 检索 3D 资产
│   ├── condition_planner.py   # 规划成功条件
│   ├── skill_planner.py       # 规划技能序列
│   ├── code_generator.py      # 生成任务代码
│   │   └── code_gen_skills/   # 代码生成子工具（entity_loader、template_registry、skill_formatter）
│   ├── registration.py        # 把生成的东西注册进系统
│   ├── reviewer.py            # 审查生成结果
│   ├── simulation.py          # 在仿真里验证
│   └── vlm_data.py            # 顺带生成 VLM 训练数据
├── tools/       # asset_cache、asset_tools、xml_injector、render_vlm_dataset、get_assets ...
└── tests/       # process_local_glb、split_glb、register_model、validate_asset（资产处理测试）
```

`docs/Node/` 下每个 node 都有一份《功能设计与实施规范》文档对应。这是项目最"工程化"的部分。

### 7. `algorithms/` & `utils/`

- `algorithms/motion_planning` + `algorithms/path_smoothing`：基于 RRT 的运动规划与轨迹平滑（配合 `src/rrt-algorithms`）。
- `utils/`：`register.py`（名称 → 类的注册表）、`utils.py`（`find_key_by_value` 等小工具）。

### 8. `assets/` — 3D 资产

```
assets/
├── scenes/   # 19 个场景：lab_0 / lab_1 / lab_copy（实验室，本分支重点）+ 各种房间
├── robots/   # 机械臂 MJCF
├── base/     # camera.xml / default.xml 等基础 XML 片段
└── review/   # 待审���资产
```

---

## 二、脚本与运行入口

| 入口 | 作用 |
|---|---|
| `scripts/download_assets.py` | 下载资产（运行前必做） |
| `scripts/trajectory_generation.py` | 生成专家演示轨迹（用 RRT 规划） |
| `scripts/trajectory_augmentation.py` | 轨迹增强 |
| `scripts/convert_to_lerobot.py` / `convert_to_rlds.py` | 转 LeRobot / RLDS 数据集格式 |
| `scripts/create_vlm_task.py` | 用 pipeline 创建 VLM 任务 |
| `scripts/evaluate_policy.py` / `evaluate_vlm.py` / `evaluate_openpi.py` | 评测入口 |
| `sh/dataset_generation.sh`、`sh/data_generation/multi_gpu_data_generation.sh` | 批量 / 多卡数据生成 |
| `sh/evaluate_openvla.sh`、`sh/evaluate_vlms.sh`、`run_openvla_eval.sh` | 批量评测 |
| `server/app.py` | 一个 Web 前端（`index.html`），可能是远程遥操 / 可视化 |
| `docker/` | 官方推荐环境（`make` 进容器，见 `docs/QuickStart.md`） |

---

## 三、与原版 VLABench 的差异

对比 `/ssd/liuzirui/VLABench`（上游主线）：

| 维度 | 原版 VLABench | 本分支 dev_LZR |
|---|---|---|
| 场景 | 厨房 / 客厅 / 麻将 / 扑克 | **实验室（lab_0 / 1）为主**，深蓝桌面 |
| 物体 | 水果 / 玩具 / 书 / 饮料 | **烧杯 / 锥形瓶 / 移液管 / 温度计 / 试管 / 漏斗** |
| 任务 | select / cluster / 复合 | **+ 大量 pick / place / pour / lift 化学原语 + 化学复合任务（拧瓶盖、加热、配液）** |
| 评测 | 主要是 openvla | **统一 policy / vlm 抽象，适配 ACT / GR00T / OpenPI / OpenVLA / RDT + 9 种 VLM** |
| 任务来源 | 手写 | **+ `pipeline/` 全自动 LLM 流水线生成** |
| 文档 | 英文 QuickStart | **大量中文设计文档、训练报告、LaTeX 论文稿（LabBench / SciVLA）** |

---

## 四、典型数据流

### 加载并运行一个任务

```
用户调 load_env("pick_pipette_...", robot="franka")
  → configs 里查 task → name2config 映射
  → register.load_robot("franka") 实例化机械臂
  → register.load_task(...) 实例化 autogen_tasks / hierarchical_tasks 里对应的 task 类
  → task 把 entities(实验器具) + scene(lab_0) + robot 组装进 dm_control Physics
  → LM4ManipDMEnv 包一层 → env.reset() / env.render() / env.step()
```

### 生成一个新任务

```
自然语言描述
  → pipeline(normalizer → analyzer → asset_manager → ... → code_generator → registration → simulation)
  → 生成 autogen_tasks/xxx_series.py + configs + 资产
  → 之后就能用 load_env 加载
```

---

## 五、环境与运行要点

- **Python 环境**：项目依赖 `dm_control` + `mujoco==3.2.2`。本机已验证可用 conda env：
  - `vlabench_openvla`（Python 3.10，mujoco 3.2.2，dm_control 已装）✅
  - `vlabench`（同上）✅
- **Jupyter kernel**：用 `vlabench_openvla` env 自带的 jupyter 启动，并把 kernel 切到 `Python (vlabench_openvla)`：
  ```bash
  export MUJOCO_GL=egl
  /ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/jupyter notebook docs/tutorials/1.load_task.ipynb
  ```
- **渲染**：无 DISPLAY 的服务器走 EGL 离屏渲染（`MUJOCO_GL=egl`），机器有 NVIDIA EGL 库即可；交互式 viewer 需要 DISPLAY 或 `xvfb-run`。
- **官方推荐**：`docker/` 下 `make` 进容器，所有依赖预装（见 `docs/QuickStart.md`）。
