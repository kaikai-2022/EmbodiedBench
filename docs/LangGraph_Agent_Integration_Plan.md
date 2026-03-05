# 🤖 VLABench LangGraph Agent 自动化流水线规划方案

## 一、整体架构设计

### 1.1 核心流程图

```
┌───────────────────────────────────────────────────────────────────┐
│                    用户自然语言输入                                │
│   "创建一个滑动盖玻片到显微镜台上的任务"                            │
└──────────────────────┬────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              LangGraph Agent Orchestrator                        │
│                   (任务编排智能体)                                │
└───────────────┬──────────────────────────────────────────────────┘
                │
    ┌───────────┴───────────┬───────────────┬─────────────┐
    │                       │               │             │
    ▼                       ▼               ▼             ▼
┌─────────┐         ┌──────────┐      ┌─────────┐    ┌─────────┐
│ Analyzer│         │ Asset    │      │ Task    │    │ Render  │
│ Node    │───────▶│ Manager  │───▶ │ Creator │──▶ │ Executor│
│ (理解)  │         │ (获取)   │      │ (创建)   │     │ (渲染)  │
└─────────┘         └──────────┘      └─────────┘    └─────────┘
    │                    │               │              │
    │                    │               │              │
    ▼                    ▼               ▼              ▼
┌─────────┐       ┌──────────┐    ┌─────────┐   ┌─────────┐
│任务需求  │       │ 模型资产 │    │env_config│  │input.png│
│解析结果  │       │ MJCF文件 │    │.json    │   │+ masks  │
└─────────┘       └──────────┘    └─────────┘   └─────────┘
```

### 1.2 Agent State 状态定义

```python
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph.message import add_messages

class VLABenchAgentState(TypedDict):
    """VLABench Agent 全局状态"""

    # 对话和任务输入
    messages: Annotated[list, add_messages]  # LLM 对话历史
    user_instruction: str                     # 用户原始自然语言指令

    # 任务分析结果 (Analyzer Node 输出)
    task_analysis: Dict[str, any]  # {
    #   "task_type": "place_coverslip_on_microscope",
    #   "objects": ["coverslip", "microscope"],
    #   "scene": "laboratory",
    #   "operation_type": "place",
    #   "instruction_parsed": "Place the coverslip onto the microscope stage"
    # }

    # 资产管理结果 (Asset Manager Node 输出)
    asset_status: Dict[str, any]  # {
    #   "coverslip": {
    #       "found": True,
    #       "xml_path": "obj/meshes/coverslip/coverslip.xml",
    #       "class": "LabEquipment"
    #   },
    #   "microscope": {
    #       "found": False,
    #       "需要下载": True,
    #       "search_keyword": "microscope"
    #   }
    # }

    # 任务配置 (Task Creator Node 输出)
    env_config: Optional[Dict]       # 完整的 env_config.json 内容
    task_save_path: Optional[str]    # 任务保存路径

    # 渲染结果 (Render Executor Node 输出)
    rendered_images: Optional[List[str]]  # 渲染图像路径列表
    validation_report: Optional[Dict]     # 场景验证报告

    # 流程控制
    current_stage: str  # 'analyzing' | 'asset_check' | 'asset_download' | 'task_creation' | 'rendering' | 'done' | 'error'
    errors: List[str]   # 错误记录
    warnings: List[str] # 警告记录
```

---

## 二、核心节点设计

### 2.1 Analyzer Node (任务理解智能体)

**职责**: 理解自然语言,提取任务关键信息

```python
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

@tool
def parse_task_instruction(instruction: str) -> dict:
    """
    解析用户自然语言指令,提取任务要素

    Args:
        instruction: 用户输入的自然语言描述

    Returns:
        {
            "objects": List[str],  # 需要的物体列表
            "scene": str,          # 场景类型 (laboratory/living_room/etc)
            "operation": str,      # 操作类型 (pick/place/pour/etc)
            "instruction_en": str, # 规范化的英文指令
            "constraints": Dict    # 额外约束条件
        }
    """
    pass

def analyzer_node(state: VLABenchAgentState):
    """
    任务分析节点 - 使用 LLM 理解用户意图
    """
    llm = ChatAnthropic(model="claude-sonnet-4-5")

    # 提示词工程
    analysis_prompt = f"""
    你是一个机器人操纵任务设计专家。用户描述了一个科研场景任务,请提取关键信息:

    用户指令: "{state['user_instruction']}"

    请分析并输出 JSON 格式:
    {{
        "objects": ["物体1", "物体2", ...],  # 需要的实验器材/物体
        "scene": "场景类型",                  # laboratory/kitchen/living_room
        "operation_type": "操作类型",         # pick/place/pour/lift/slide
        "instruction_en": "规范化英文指令",
        "spatial_relations": ["关系描述"],    # 如 "coverslip on microscope stage"
        "task_name": "任务名称(英文下划线)",  # 如 place_coverslip_on_microscope
    }}

    注意:
    - 识别科研器材的专业名称 (如 coverslip=盖玻片, microscope=显微镜)
    - 推断合理的操作类型
    - 生成符合 VLABench 规范的任务名称
    """

    response = llm.invoke(analysis_prompt)
    task_analysis = json.loads(response.content)

    return {
        "task_analysis": task_analysis,
        "current_stage": "asset_check",
        "messages": [response]
    }
```

---

### 2.2 Asset Manager Node (资产管理智能体)

**职责**: 检查所需模型资产,自动调用 get_assets.py 下载缺失模型

```python
@tool
def check_asset_exists(object_name: str) -> dict:
    """
    检查指定物体的 MJCF 资产是否存在

    Args:
        object_name: 物体名称 (如 "microscope")

    Returns:
        {
            "found": bool,
            "xml_path": str or None,
            "class": str or None
        }
    """
    from VLABench.configs.constant import name2class_xml

    # 在 name2class_xml 中查找
    if object_name in name2class_xml:
        class_type, xml_path = name2class_xml[object_name]
        return {
            "found": True,
            "xml_path": xml_path if isinstance(xml_path, str) else xml_path[0],
            "class": class_type.__name__
        }

    # 在文件系统中搜索
    asset_dir = Path(os.environ['VLABENCH_ROOT']) / 'assets' / 'obj' / 'meshes'
    matches = list(asset_dir.glob(f"**/*{object_name}*.xml"))

    if matches:
        return {
            "found": True,
            "xml_path": str(matches[0].relative_to(asset_dir.parent)),
            "class": "GenericObject"
        }

    return {"found": False, "xml_path": None, "class": None}

@tool
def download_asset(keyword: str, max_downloads: int = 3) -> dict:
    """
    调用 get_assets.py 下载指定关键词的模型

    Args:
        keyword: 搜索关键词 (如 "microscope")
        max_downloads: 最大下载数量

    Returns:
        {
            "success": bool,
            "downloaded_count": int,
            "assets": List[str]  # 下载的资产路径列表
        }
    """
    import subprocess

    result = subprocess.run([
        "python", "scripts/get_assets.py",
        "--keyword", keyword,
        "--max_downloads", str(max_downloads),
        "--output_dir", "./assets/review",
        "--skip_existing"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        # 解析输出,提取下载的资产信息
        output_dir = Path("./assets/review") / keyword
        assets = list(output_dir.glob("*/*.xml"))

        return {
            "success": True,
            "downloaded_count": len(assets),
            "assets": [str(a) for a in assets]
        }
    else:
        return {
            "success": False,
            "downloaded_count": 0,
            "assets": [],
            "error": result.stderr
        }

def asset_manager_node(state: VLABenchAgentState):
    """
    资产管理节点 - 检查并下载所需模型
    """
    task_analysis = state['task_analysis']
    required_objects = task_analysis['objects']

    asset_status = {}
    missing_objects = []

    # 1. 检查所有物体资产
    for obj in required_objects:
        status = check_asset_exists(obj)
        asset_status[obj] = status

        if not status['found']:
            missing_objects.append(obj)

    # 2. 下载缺失的资产
    if missing_objects:
        print(f"缺失资产: {missing_objects}, 开始下载...")

        for obj in missing_objects:
            download_result = download_asset(obj)

            if download_result['success'] and download_result['assets']:
                # 更新资产状态
                asset_status[obj] = {
                    "found": True,
                    "xml_path": download_result['assets'][0],
                    "class": "DownloadedAsset",
                    "newly_downloaded": True
                }
            else:
                # 下载失败,记录错误
                state['errors'].append(f"无法下载资产: {obj}")

    return {
        "asset_status": asset_status,
        "current_stage": "task_creation" if not state['errors'] else "error"
    }
```

---

### 2.3 Task Creator Node (任务生成智能体)

**职责**: 基于分析结果和资产状态,生成 env_config.json 和任务目录

```python
@tool
def generate_env_config(
    task_name: str,
    objects: List[Dict],
    scene: str,
    instruction: str,
    operation_sequence: List[Dict],
    conditions: Dict
) -> Dict:
    """
    生成 VLABench env_config.json

    Args:
        task_name: 任务名称
        objects: 物体列表,每个包含 {name, xml_path, class, position, orientation}
        scene: 场景名称
        instruction: 任务指令
        operation_sequence: 操作序列
        conditions: 评测条件

    Returns:
        完整的 env_config dict
    """
    # 基础配置
    env_config = {
        "task": {
            "components": [
                {
                    "name": "table",
                    "xml_path": "obj/meshes/table/table.xml",
                    "position": [0, 0, 0],
                    "orientation": [1, 0, 0, 0],
                    "class": "Table",
                    "materials": ["wood3"]
                }
            ],
            "scene": {
                "name": scene,
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "floor_textures": ["lab_floor"]
            },
            "instructions": [instruction],
            "conditions": conditions
        }
    }

    # 添加任务物体
    for obj in objects:
        env_config["task"]["components"].append({
            "name": obj["name"],
            "xml_path": obj["xml_path"],
            "position": obj.get("position", [0.0, 0.0, 0.8]),
            "orientation": obj.get("orientation", [1, 0, 0, 0]),
            "class": obj["class"]
        })

    return env_config

def task_creator_node(state: VLABenchAgentState):
    """
    任务创建节点 - 生成 env_config 和任务目录结构
    """
    task_analysis = state['task_analysis']
    asset_status = state['asset_status']

    # 1. 构建物体列表
    objects = []
    for i, obj_name in enumerate(task_analysis['objects']):
        asset_info = asset_status[obj_name]

        # 生成随机位置(避免碰撞)
        position = [
            random.uniform(-0.2, 0.2),
            random.uniform(-0.2, 0.2),
            0.8  # 桌面高度
        ]

        objects.append({
            "name": f"{obj_name}_{i}",
            "xml_path": asset_info['xml_path'],
            "class": asset_info['class'],
            "position": position,
            "orientation": [1, 0, 0, 0]
        })

    # 2. 推断操作序列 (使用 LLM)
    llm = ChatAnthropic(model="claude-sonnet-4-5")

    operation_prompt = f"""
    任务: {task_analysis['instruction_en']}
    物体: {[o['name'] for o in objects]}
    操作类型: {task_analysis['operation_type']}

    生成 VLABench 操作序列 JSON:
    [
        {{"name": "pick", "params": {{"target_entity_name": <物体索引>}}}},
        {{"name": "place/pour/lift", "params": {{...}}}}
    ]

    物体索引从 1 开始 (0 是桌子)
    """

    op_response = llm.invoke(operation_prompt)
    operation_sequence = json.loads(op_response.content)

    # 3. 生成评测条件
    conditions = {}
    if task_analysis['operation_type'] == 'pour':
        conditions = {
            "pour": {
                "target_entity": objects[0]["name"],
                "threshold": 0
            }
        }
    elif task_analysis['operation_type'] == 'place':
        conditions = {
            "contain": {
                "container": objects[1]["name"],
                "entities": [objects[0]["name"]]
            }
        }

    # 4. 生成 env_config
    env_config = generate_env_config(
        task_name=task_analysis['task_name'],
        objects=objects,
        scene=task_analysis['scene'] + "_0",
        instruction=task_analysis['instruction_en'],
        operation_sequence=operation_sequence,
        conditions=conditions
    )

    # 5. 创建任务目录结构
    task_dir = DATASET_ROOT / "M&T" / task_analysis['task_name'] / "example0"
    task_dir.mkdir(parents=True, exist_ok=True)

    # 保存 env_config.json
    config_dir = task_dir / "env_config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "env_config.json"

    with open(config_path, 'w') as f:
        json.dump(env_config, f, indent=4)

    # 保存 instruction.txt
    input_dir = task_dir / "input"
    input_dir.mkdir(exist_ok=True)
    (input_dir / "instruction.txt").write_text(task_analysis['instruction_en'])

    # 保存 operation_sequence.json
    output_dir = task_dir / "output"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "operation_sequence.json", 'w') as f:
        json.dump({"skill_sequence": operation_sequence}, f, indent=4)

    return {
        "env_config": env_config,
        "task_save_path": str(task_dir),
        "current_stage": "rendering"
    }
```

---

### 2.4 Render Executor Node (渲染执行智能体)

**职责**: 调用 render_vlm_dataset.py 生成图像并验证场景

```python
@tool
def render_vlm_task(task_name: str, dimension: str = "M&T") -> dict:
    """
    调用 render_vlm_dataset.py 渲染任务场景

    Args:
        task_name: 任务名称
        dimension: 评测维度

    Returns:
        {
            "success": bool,
            "rendered_images": List[str],
            "validation_report": Dict
        }
    """
    import subprocess

    result = subprocess.run([
        "python", "scripts/render_vlm_dataset.py",
        "--task", task_name,
        "--dimension", dimension,
        "--num-examples", "1"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        # 查找渲染的图像
        task_dir = DATASET_ROOT / dimension / task_name / "example0" / "input"
        images = list(task_dir.glob("*.png"))

        # 读取验证报告
        validation_path = task_dir.parent / "env_config" / "validation_report.json"
        validation_report = {}
        if validation_path.exists():
            with open(validation_path) as f:
                validation_report = json.load(f)

        return {
            "success": True,
            "rendered_images": [str(img) for img in images],
            "validation_report": validation_report
        }
    else:
        return {
            "success": False,
            "rendered_images": [],
            "validation_report": {},
            "error": result.stderr
        }

def render_executor_node(state: VLABenchAgentState):
    """
    渲染执行节点 - 调用渲染脚本并验证
    """
    task_name = state['task_analysis']['task_name']

    # 1. 渲染场景
    render_result = render_vlm_task(task_name)

    if not render_result['success']:
        return {
            "current_stage": "error",
            "errors": state['errors'] + [f"渲染失败: {render_result.get('error', 'Unknown')}"]
        }

    # 2. 检查验证报告
    validation = render_result['validation_report']
    warnings = []

    if validation.get('overall_status') != 'PASS':
        warnings.append("场景验证未通过,请检查验证报告")
        for error in validation.get('errors', []):
            warnings.append(f"  - {error}")

    return {
        "rendered_images": render_result['rendered_images'],
        "validation_report": validation,
        "warnings": state['warnings'] + warnings,
        "current_stage": "done"
    }
```

---

## 三、LangGraph 图构建

### 3.1 完整图结构

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def build_vlabench_agent():
    """构建 VLABench 自动化流水线 Agent"""

    # 创建图
    workflow = StateGraph(state_schema=VLABenchAgentState)

    # 添加节点
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("asset_manager", asset_manager_node)
    workflow.add_node("task_creator", task_creator_node)
    workflow.add_node("render_executor", render_executor_node)
    workflow.add_node("error_handler", error_handler_node)

    # 添加边
    workflow.add_edge(START, "analyzer")

    # 条件边: analyzer -> asset_manager
    workflow.add_edge("analyzer", "asset_manager")

    # 条件边: asset_manager -> task_creator (成功) 或 error (失败)
    workflow.add_conditional_edges(
        "asset_manager",
        lambda state: "task_creator" if state['current_stage'] != "error" else "error",
        {
            "task_creator": "task_creator",
            "error": "error_handler"
        }
    )

    # task_creator -> render_executor
    workflow.add_edge("task_creator", "render_executor")

    # render_executor -> END
    workflow.add_edge("render_executor", END)
    workflow.add_edge("error_handler", END)

    # 编译图 (添加 checkpointer 用于状态持久化)
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph

def error_handler_node(state: VLABenchAgentState):
    """错误处理节点"""
    error_msg = "\n".join(state['errors'])
    print(f"❌ Agent 执行失败:\n{error_msg}")

    return {
        "current_stage": "error",
        "messages": [{"role": "assistant", "content": f"任务失败: {error_msg}"}]
    }
```

---

## 四、使用接口

### 4.1 命令行接口

```python
# vlabench_agent_cli.py
import argparse
from vlabench_agent import build_vlabench_agent

def main():
    parser = argparse.ArgumentParser(description="VLABench 自然语言任务生成 Agent")
    parser.add_argument(
        '--instruction',
        type=str,
        required=True,
        help='任务描述 (自然语言)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='可选: 提供 YAML 配置文件路径'
    )

    args = parser.parse_args()

    # 初始化 Agent
    agent = build_vlabench_agent()

    # 初始状态
    initial_state = {
        "messages": [],
        "user_instruction": args.instruction,
        "task_analysis": {},
        "asset_status": {},
        "env_config": None,
        "task_save_path": None,
        "rendered_images": None,
        "validation_report": None,
        "current_stage": "analyzing",
        "errors": [],
        "warnings": []
    }

    # 运行 Agent
    print("🤖 VLABench Agent 启动...")
    print(f"📝 任务: {args.instruction}\n")

    result = agent.invoke(initial_state)

    # 输出结果
    if result['current_stage'] == 'done':
        print("✅ 任务完成!")
        print(f"📁 保存路径: {result['task_save_path']}")
        print(f"🖼️  渲染图像: {result['rendered_images']}")

        if result['warnings']:
            print("\n⚠️  警告:")
            for warn in result['warnings']:
                print(f"  - {warn}")
    else:
        print("❌ 任务失败")
        for err in result['errors']:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
```

**使用示例**:
```bash
python vlabench_agent_cli.py --instruction "创建一个滑动盖玻片到显微镜台上的任务"

# 输出:
# 🤖 VLABench Agent 启动...
# 📝 任务: 创建一个滑动盖玻片到显微镜台上的任务
#
# [Analyzer] 正在分析任务...
# [Analyzer] 识别到物体: ['coverslip', 'microscope']
# [Analyzer] 操作类型: place
#
# [Asset Manager] 检查资产...
# [Asset Manager] coverslip: 未找到,开始下载...
# [Asset Manager] microscope: 未找到,开始下载...
# [Asset Manager] 下载完成: 2 个资产
#
# [Task Creator] 生成任务配置...
# [Task Creator] 任务名称: place_coverslip_on_microscope
# [Task Creator] env_config.json 已生成
#
# [Render Executor] 渲染场景...
# [Render Executor] 场景验证: PASS
#
# ✅ 任务完成!
# 📁 保存路径: /path/to/dataset/vlm_evaluation_v1.0/M&T/place_coverslip_on_microscope/example0
# 🖼️  渲染图像: ['input.png', 'input_mask.png']
```

---

## 五、项目集成

### 5.1 目录结构

```
VLABench/
├── scripts/
│   ├── get_assets.py               # (已有) 资产下载
│   ├── create_vlm_task.py          # (已有) 任务创建
│   ├── render_vlm_dataset.py       # (已有) 渲染脚本
│   └── vlabench_agent/             # (新增) Agent 目录
│       ├── __init__.py
│       ├── agent.py                # Agent 主逻辑
│       ├── nodes/                  # 节点实现
│       │   ├── __init__.py
│       │   ├── analyzer.py
│       │   ├── asset_manager.py
│       │   ├── task_creator.py
│       │   └── render_executor.py
│       ├── tools/                  # 工具函数
│       │   ├── __init__.py
│       │   ├── asset_tools.py
│       │   └── task_tools.py
│       └── prompts/                # 提示词模板
│           ├── analyzer_prompt.txt
│           └── operation_prompt.txt
├── vlabench_agent_cli.py           # (新增) CLI 入口
├── requirements_agent.txt          # (新增) Agent 依赖
└── README_AGENT.md                 # (新增) Agent 文档
```

### 5.2 依赖安装

```txt
# requirements_agent.txt
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.2.0  # 使用 Claude Sonnet 4.5
langchain-core>=0.3.0
pydantic>=2.0.0
```

```bash
pip install -r requirements_agent.txt
```

---

## 六、高级特性

### 6.1 人工审核节点 (Human-in-the-Loop)

```python
from langgraph.prebuilt import interrupt

def human_review_node(state: VLABenchAgentState):
    """人工审核节点 - 在关键步骤暂停等待用户确认"""

    # 展示任务配置
    print("\n📋 任务配置预览:")
    print(json.dumps(state['env_config'], indent=2))

    # 中断流程,等待人工确认
    user_input = interrupt({
        "message": "请审核以上配置,是否继续?",
        "options": ["继续", "修改", "取消"]
    })

    if user_input == "取消":
        return {
            "current_stage": "error",
            "errors": ["用户取消任务"]
        }
    elif user_input == "修改":
        # 回退到 task_creator 重新生成
        return {"current_stage": "task_creation"}
    else:
        return {"current_stage": "rendering"}
```

### 6.2 多任务批量生成

```python
def batch_generate_tasks(instructions: List[str]) -> List[Dict]:
    """批量生成多个任务"""
    agent = build_vlabench_agent()
    results = []

    for instruction in instructions:
        initial_state = create_initial_state(instruction)
        result = agent.invoke(initial_state)
        results.append(result)

    return results

# 使用示例
instructions = [
    "创建一个滑动盖玻片到显微镜台上的任务",
    "创建一个将试管放入试管架的任务",
    "创建一个倾倒烧杯的任务"
]

results = batch_generate_tasks(instructions)
```

### 6.3 配置验证与优化

```python
def optimization_node(state: VLABenchAgentState):
    """
    配置优化节点 - 基于验证报告自动调整物体位置
    """
    validation = state['validation_report']

    if validation.get('overall_status') == 'FAILED':
        # 提取错误信息
        errors = validation.get('errors', [])

        # 使用 LLM 生成修复方案
        llm = ChatAnthropic(model="claude-sonnet-4-5")

        fix_prompt = f"""
        场景验证失败,错误如下:
        {json.dumps(errors, indent=2)}

        当前 env_config:
        {json.dumps(state['env_config'], indent=2)}

        请生成修复后的 env_config (调整物体位置/朝向),确保:
        1. 物体不重叠
        2. 物体在相机视野内
        3. 物体放置合理

        只返回修复后的 JSON。
        """

        response = llm.invoke(fix_prompt)
        fixed_config = json.loads(response.content)

        # 重新渲染
        return {
            "env_config": fixed_config,
            "current_stage": "rendering"
        }

    return state
```

---

## 七、与现有脚本的对比

### 7.1 对话中建议的局限性

对话中提到的方案存在以下问题:

1. **过度简化**: 直接用 `subprocess` 调用脚本,没有充分利用 VLABench 的内部 API
2. **状态管理缺失**: 没有设计完整的 State Schema
3. **缺乏智能**: 没有利用 LLM 的理解和推理能力
4. **错误处理弱**: 没有设计自动重试和修复机制

### 7.2 本方案的优势

| 维度 | 对话方案 | 本方案 |
|------|---------|--------|
| **任务理解** | 结构化参数输入 | 自然语言理解 + LLM 推理 |
| **资产管理** | 手动指定路径 | 自动检测 + 自动下载 |
| **配置生成** | 基于模板填充 | LLM 智能生成 + 验证优化 |
| **错误处理** | 简单 try-catch | 自动修复 + 人工介入 |
| **可扩展性** | 难以扩展 | 节点化设计,易于添加新功能 |
| **用户体验** | 需要理解配置格式 | 纯自然语言交互 |

---

## 八、实施路线图

### Phase 1: 基础流水线 (1-2 周)
- [ ] 实现 Analyzer Node (基于 Claude Sonnet 4.5)
- [ ] 实现 Asset Manager Node (集成 get_assets.py)
- [ ] 实现 Task Creator Node (基于模板)
- [ ] 实现 Render Executor Node (调用 render_vlm_dataset.py)
- [ ] 构建 LangGraph 图
- [ ] CLI 接口

### Phase 2: 智能优化 (1 周)
- [ ] 添加场景验证反馈循环
- [ ] 实现配置自动修复
- [ ] 添加人工审核节点 (Human-in-the-Loop)

### Phase 3: 高级特性 (1-2 周)
- [ ] 批量任务生成
- [ ] 任务模板库扩展
- [ ] Web UI 界面 (可选)
- [ ] 多 Agent 协作 (可选)

---

## 九、测试用例

```python
# test_vlabench_agent.py
def test_simple_task():
    """测试简单任务生成"""
    instruction = "创建一个倾倒试管的任务"
    agent = build_vlabench_agent()

    result = agent.invoke(create_initial_state(instruction))

    assert result['current_stage'] == 'done'
    assert result['task_save_path'] is not None
    assert len(result['rendered_images']) > 0

def test_complex_task():
    """测试复杂任务生成(需要下载资产)"""
    instruction = "创建一个将盖玻片放在显微镜台上的任务"
    agent = build_vlabench_agent()

    result = agent.invoke(create_initial_state(instruction))

    # 验证资产下载
    assert 'coverslip' in result['asset_status']
    assert result['asset_status']['coverslip']['found']

    # 验证任务生成
    assert result['env_config'] is not None
    assert 'coverslip' in result['env_config']['task']['instructions'][0]
```

---

## 十、总结

这个方案相比对话中的建议:

✅ **更智能**: 使用 LLM 理解自然语言,无需手动配置
✅ **更自动**: 自动检测和下载缺失资产
✅ **更健壮**: 内置验证和自动修复机制
✅ **更灵活**: 节点化设计,易于扩展新功能
✅ **更符合项目实际**: 深度集成现有脚本,而非简单调用

关键区别在于: **这不是简单的脚本串联,而是一个智能任务编排系统**。
