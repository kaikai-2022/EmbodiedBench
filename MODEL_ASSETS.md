# VLAbench_LZR 模型资产说明文档

## 概述

VLAbench_LZR 是一个视觉-语言-动作（VLA）模型的评估框架，包含了多种策略模型和视觉语言模型的实现。本文档详细记录了项目中所有模型资产的配置、来源和使用方式。

## 目录结构

```
/ssd/liuzirui/VLAbench_LZR/
├── VLABench/
│   ├── configs/model/            # 模型配置文件
│   │   └── openvla_config.json   # OpenVLA 标准化配置
│   ├── evaluation/model/         # 模型实现代码
│   │   ├── policy/               # 策略模型（VLA）
│   │   │   ├── openvla.py        # OpenVLA
│   │   │   ├── act.py            # ACT
│   │   │   ├── gr00t.py          # Gr00t
│   │   │   ├── openpi.py         # OpenPi
│   │   │   └── base.py           # 基类
│   │   └── vlm/                  # 视觉语言模型
│   │       ├── gpt4.py           # GPT-4v
│   │       ├── qwen_vl.py        # Qwen2-VL
│   │       ├── llava.py          # LLaVA-NeXT
│   │       ├── .py            # GLM-4v
│   │       ├── intern_vl.py      # InternVL2
│   │       ├── minicpm.py        # MiniCPM-V
│   │       ├── claude.py         # Claude
│   │       └── gemini.py         # Gemini
│   └── pipeline/config.py        # Agent配置
├── scripts/
│   ├── evaluate_policy.py        # 策略模型评估脚本
│   └── evaluate_vlm.py           # VLM评估脚本
└── sh/                           # Shell脚本
```

## 一、策略模型（VLA / Policy Models）

### 1.1 OpenVLA

**实现文件**: `VLABench/evaluation/model/policy/openvla.py`

**模型信息**:
- **HuggingFace ID**: `openvla/openvla-7b`
- **基础LLM**: `meta-llama/Llama-2-7b-hf`
- **模型类型**: Vision-Language-Action模型
- **架构**: Llama2 + Vision Encoder + Action Head

**配置参数**:
```python
{
    "n_action_bins": 256,           # 动作离散化bin数量
    "image_sizes": [224, 224],     # 输入图像尺寸
    "llm_max_length": 2048,        # LLM最大序列长度
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32
}
```

**加载方式**:
```python
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
```

**微调支持**:
- 支持LoRA权重加载
- 默认微调权重路径: `/remote-home1/pjliu/openvla/weights/vlabench/select_fruit+CSv1+lora/`

---

### 1.2 ACT (Action Chunking Transformer)

**实现文件**: `VLABench/evaluation/model/policy/act.py`

**模型信息**:
- **基础框架**: LeRobot
- **框架路径**: `/ssd/liuzirui/lerobot`
- **模型类**: `LeRobotACTPolicy`
- **视觉骨干**: ResNet18 (ImageNet预训练)

**配置参数**:
```python
{
    "input_dim": 4,                # 输入维度 (图像+状态)
    "hidden_dim": 512,             # 隐藏层维度
    "num_layers": 4,               # Transformer层数
    "num_heads": 8,                # 注意力头数
    "chunk_size": 100,             # 动作块大小
    "dropout": 0.1
}
```

**特点**:
- 离线训练的模仿学习模型
- 需要特定格式的训练数据
- 支持多种机器人平台

---

### 1.3 Gr00t

**实现文件**: `VLABench/evaluation/model/policy/gr00t.py`

**模型信息**:
- **架构**: 远程推理服务
- **通信协议**: ZMQ (TCP)
- **服务端点**: `tcp://localhost:5555`

**工作方式**:
- 本地作为ZMQ客户端
- 连接到远程Gr00t推理服务
- 发送图像和状态，接收动作预测

**配置要求**:
- 需要单独启动Gr00t推理服务
- 默认端口: 5555
- 支持批处理推理

---

### 1.4 OpenPi

**实现文件**: `VLABench/evaluation/model/policy/openpi.py`

**模型信息**:
- **架构**: 远程推理客户端
- **参考实现**: `third_party/openpi`
- **基础**: OpenVLA变体

**特点**:
- 优化的推理框架
- 支持实时控制
- 需要额外依赖安装

---

## 二、视觉语言模型（VLM Models）

### 2.1 GPT-4v (OpenAI)

**实现文件**: `VLABench/evaluation/model/vlm/gpt4.py`

**模型信息**:
- **提供商**: OpenAI
- **模型名称**: `gpt-4-vision-preview`
- **API端点**: `https://api.openai.com/v1`

**配置要求**:
```python
# 在 VLABench/pipeline/config.py 中配置
os.environ["OPENAI_API_KEY"] = "your-api-key"
# 或使用自定义端点
API_BASE = "https://vip.aipro.love"
```

**使用限制**:
- 需要有效的OpenAI API密钥
- 按调用次数计费
- 支持高分辨率图像输入

---

### 2.2 Qwen2-VL (通义千问)

**实现文件**: `VLABench/evaluation/model/vlm/qwen_vl.py`

**模型信息**:
- **提供商**: 阿里云
- **ModelScope ID**: `/2-VL-7B-Instruct`
- **框架**: ModelScope / transformers

**配置参数**:
```python
{
    "model_path": "Qwen/Qwen2-VL-7B-Instruct",
    "max_image_pixels": 1280 * 28 * 28,
    "min_pixels": 4 * 16 * 16,
    "max_dynamic_patches": 8
}
```

**加载方式**:
```python
from modelscope import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto"
)
```

---

### 2.3 LLaVA-NeXT

**实现文件**: `VLABench/evaluation/model/vlm/llava.py`

**模型信息**:
- **HuggingFace ID**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **基础LLM**: Mistral-7B
- **架构**: LLaVA-1.6

**配置参数**:
```python
{
    "model_path": "llava-hf/llava-v1.6-mistral-7b-hf",
    "image_size": 336,
    "max_tokens": 2048
}
```

**加载方式**:
```python
from transformers import LlavaNextForConditionalGeneration
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
```

---

### 2.4 GLM-4v

**实现文件**: `VLABench/evaluation/model/vlm/glm.py`

**模型信息**:
- **提供商**: 智谱AI
- **HuggingFace ID**: `THUDM/glm-4v-9b`
- **架构**: GLM-4

**配置参数**:
```python
{
    "model_path": "THUDM/glm-4v-9b",
    "trust_remote_code": True
}
```

---

### 2.5 InternVL2

**实现文件**: `VLABench/evaluation/model/vlm/intern_vl.py`

**模型信息**:
- **提供商**: 上海人工智能实验室
- **HuggingFace ID**: `OpenGVLab/InternVL2-8B`
- **推理框架**: lmdeploy

**配置要求**:
```bash
# 安装lmdeploy
pip install lmdeploy
```

**特点**:
- 支持高效推理
- 需要lmdeploy依赖
- 支持批处理

---

### 2.6 MiniCPM-V

**实现文件**: `VLABench/evaluation/model/vlm/minicpm.py`

**模型信息**:
- **HuggingFace ID**: `openbmb/MiniCPM-V-2_6`
- **推理框架**: vLLM

**配置参数**:
```python
{
    "model_path": "openbmb/MiniCPM-V-2_6",
    "trust_remote_code": True
}
```

**特点**:
- 轻量级模型
- 适合边缘部署
- 需要vLLM支持

---

### 2.7 Claude (Anthropic)

**实现文件**: `VLABench/evaluation/model/vlm/claude.py`

**模型信息**:
- **提供商**: Anthropic
- **默认模型**: `claude-3-7-sonnet-20250219`
- **可用模型**: 
  - `claude-3-7-sonnet-20250219`
  - `claude-3-5-sonnet-20241022`
  - `claude-3-opus-20240229

**配置要求**:
```python
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
```

---

### 2.8 Gemini (Google)

**实现文件**: `VLABench/evaluation/model/vlm/gemini.py`

**模型信息**:
- **提供商**: Google
- **默认模型**: `gemini-2.5-pro-exp-03-25`
- **可用模型**:
  - `gemini-2.5-pro-exp-03-25`
  - `gemini-2.0-flash-exp`
  - `gemini-1.5-pro`

**配置要求**:
```python
import google.generativeai as genai
genai.configure(api_key="your-api-key")
```

---

## 三、配置文件详解

### 3.1 OpenVLA 标准化配置

**文件路径**: `VLABench/configs/model/openvla_config.json`

**文件大小**: 71,910 bytes

**功能**: 存储OpenVLA模型在各个预训练数据集上的action和proprioception统计信息，用于模型输出的标准化。

**包含的数据集** (26个):

1. `austin_buds_dataset` - Austin Buds机器人数据
2. `austin_sailor_dataset` - Austin Sailor机器人数据
3. `austin_sirius_dataset` - Austin Sirius机器人数据
4. `bc_z` - Berkeley Cable数据集
5. `berkeley_autolab_ur5` - Berkeley AutoLab UR5数据
6. `berkeley_cable_routing` - Berkeley线缆路由数据
7. `berkeley_fanuc_manipulation` - Berkeley Fanuc操作数据
8. `bridge_orig` - Bridge数据集原始版本
9. `cmu_stretch` - CMU Stretch机器人数据
10. `dlr_edan_shared_control` - DLR EDAN共享控制数据
11. `dobbe` - Dobbe数据集
12. `fmb_dataset` - FMB数据集
13. `fractal20220817_data` - Fractal数据集
14. `furniture_bench_dataset` - 家具组装数据集
15. `iamlab_cmu_pickup_insert` - CMU拾取插入数据
16. `jaco_play` - Jaco操作数据
17. `kuka` - Kuka机器人数据
18. `nyu_franka_play_dataset` - NYU Franka数据
19. `roboturk` - RoboTurk数据集
20. `stanford_hydra_dataset` - Stanford Hydra数据
21. `taco_play` - TACO数据集
22. `toto` - TOTO数据集
23. `ucsd_kitchen_dataset` - UCSD厨房数据
24. `utaustin_mutex` - UT Austin Mutex数据
25. `viola` - Viola数据集
26. `select_toy` - 玩具选择数据

**标准化参数结构**:
```json
{
    "dataset_name": {
        "action": {
            "mean": [...],  # 动作均值向量
            "std": [...]    # 动作标准差向量
        },
        "proprio": {
            "mean": [...],  # 本体感受均值向量
            "std": [...]    # 本体感受标准差向量
        }
    }
}
```

---

## 四、模型资产统计汇总

| 资产类型 | 数量 | 本地文件 | 说明 |
|---------|------|---------|------|
| **本地配置文件** | 1 | openvla_config.json (70KB) | OpenVLA标准化配置 |
| **VLA模型实现** | 4 | - | OpenVLA, ACT, Gr00t, OpenPi |
| **VLM模型实现** | 8 | - | GPT-4v, Qwen2-VL, LLaVA, GLM-4v, InternVL2, MiniCPM-V, Claude, Gemini |
| **预训练模型权重** | 0 | 0 | 所有模型从远程动态下载 |
| **LoRA权重** | 0 | 0 | 需单独下载微调权重 |

**本地模型文件大小**: ~70KB (仅配置文件)

**远程模型大小** (需要下载):
- OpenVLA-7B: ~7GB
- Qwen2-VL-7B: ~14GB
- LLaVA-NeXT-7B: ~14GB
- GLM-4v-9B: ~18GB
- InternVL2-8B: ~16GB
- MiniCPM-V-2.6: ~8GB

**总计**: 约77GB (如果下载所有模型)

---

## 五、API密钥配置

### 5.1 OpenAI (GPT-4v)

```bash
export OPENAI_API_KEY="sk-..."
# 或使用自定义端点
export OPENAI_API_BASE="https://vip.aipro.love"
```

### 5.2 Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 5.3 Google (Gemini)

```bash
export GEMINI_API_KEY="..."
```

### 5.4 配置文件位置

`VLABench/pipeline/config.py`

---

## 六、第三方框架依赖

| 框架 | 用途 | 本地路径 |
|------|------|---------|
| LeRobot | ACT模型 | `/ssd/liuzirui/lerobot` |
| lmdeploy | InternVL2推理 | 需pip安装 |
| vLLM | MiniCPM-V推理 | 需pip安装 |
| ModelScope | Qwen2-VL加载 | 需pip安装 |
| OpenPI | OpenPi实现 | `third_party/openpi` |

---

## 七、模型下载和使用指南

### 7.1 下载评估数据

```bash
# 下载VLM评估数据
git clone https://huggingface.co/datasets/VLABench/vlm_evaluation_v1.0

# 下载微调数据集
git clone https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_dataset
```

### 7.2 模型自动下载

所有模型在首次使用时会自动从HuggingFace或ModelScope下载:

```python
# OpenVLA
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")

# Qwen2-VL
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# LLaVA
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
```

### 7.3 缓存位置

默认HuggingFace缓存: `~/.cache/huggingface/hub/`

---

## 八、评估脚本使用

### 8.1 评估策略模型

```bash
python scripts/evaluate_policy.py \
    --model_ckpt /path/to/model \
    --lora_ckpt /path/to/lora \
    --dataset_name <dataset> \
    --epoch 100
```

### 8.2 评估VLM模型

```bash
python scripts/evaluate_vlm.py \
    --model_name gpt4 \
    --api_key <your_key> \
    --dataset_name <dataset>
```

---

## 九、注意事项

1. **无本地模型权重**: 项目不包含任何预训练模型权重文件，所有模型需要从远程下载或通过API访问

2. **API费用**: GPT-4v、Claude、Gemini需要配置API密钥并按调用次数付费

3. **磁盘空间**: 完整下载所有模型需要约77GB磁盘空间

4. **GPU要求**: 
   - 7B模型需要至少16GB GPU内存
   - 9B模型需要至少24GB GPU内存

5. **网络要求**: 首次下载模型需要稳定网络连接

---

## 十、更新日志

- **2025-07-07**: 初始版本，记录所有模型资产配置

---

## 联系方式

如有问题，请联系项目维护者。
