# 环境修复说明（`test_e2e.py` 跑通的踩坑记录）

> 日期：2026-07-03
> 触发的命令：
> ```bash
> python /ssd/liuzirui/VLAbench_LZR/VLABench/pipeline/tests/test_e2e.py "place florence_flask on the tripod"
> ```
> 最终结果：✅ **7 节点 pipeline 全通，仿真成功，生成视频**（`/ssd/liuzirui/VLAbench_LZR/dataset/autogen_tasks/place_florence_flask/demo_0_success_True.mp4`，12 秒 / 960×960 / h264 / 451 KB）

本文记录从"完全跑不起来"到"端到端跑通 + 视频保存"的全过程。

---

## 0. 根因速览

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: VLABench` | 10 个 `.py` 里有原作者的硬编码路径 `/ssd/mkqin/workspace/VLABench` | 全部改成 `/ssd/liuzirui/VLAbench_LZR`（直接硬编码替换） |
| 2 | `ModuleNotFoundError: No module named 'langgraph'` | pipeline 依赖 `langgraph`，`vlabench_openvla` env 没装 | `pip install langgraph`（含 langchain-core / langgraph-checkpoint 等传递依赖） |
| 3 | `FileNotFoundError: .../obj/meshes/containers/basket`（import 期崩） | `VLABench/assets/obj/` 没下载；`configs/constant.py` 在 import 时无差别 `os.listdir`，目录缺失就崩 | 把 `assets/obj` 软链接到原版 `/ssd/liuzirui/VLABench/VLABench/assets/obj`（资产复用） |
| 4 | `ERROR ... langchain_anthropic 未安装` | pipeline 7 节点里 `analyzer/normalizer/reviewer` 都要调 `ChatAnthropic` | `pip install langchain_anthropic anthropic`；config.py 里 key/base_url 已被前任写好，可用第三方代理 |
| 5 | `Program 'ffmpeg' is not found`（`mediapy.write_video` 抛） | 系统无 ffmpeg，且当前用户 `liuzirui` 无 sudo 权限，不能 `apt install ffmpeg` | 把 `imageio-ffmpeg` 自带的 ffmpeg 二进制软链接到 `~/.local/bin/ffmpeg`（PATH 里，无需 sudo） |
| 6 | 每次新 shell `python` 又是 base 的 3.13，报 `dm_control` 找不到 | `conda 26.3.2` 要求先 `source conda.sh`，且默认 env 是 base | `.bashrc` 末尾追加 `conda activate vlabench_openvla`，新 shell 自动激活（见第 9 节） |

---

## 1. 硬编码路径替换

仓库 fork 自原作者 `mkqin`，`VLABench/`、`scripts/` 下 10 个 `.py` 写死了 `/ssd/mkqin/workspace/VLABench`、`/ssd/mkqin/workspace/lerobot`、`source /ssd/mkqin/miniconda3/...` 等路径。本机没有这些目录。

**做法**（按您选择"直接硬编码替换"）：

```bash
cd /ssd/liuzirui/VLAbench_LZR

# VLABench 仓库根
find . -name "*.py" -not -path "./.git/*" -exec sed -i \
  's|/ssd/mkqin/workspace/VLABench|/ssd/liuzirui/VLAbench_LZR|g' {} +

# lerobot 目录
find . -name "*.py" -not -path "./.git/*" -exec sed -i \
  's|/ssd/mkqin/workspace/lerobot|/ssd/liuzirui/lerobot|g' {} +

# conda 激活命令
find . -name "*.py" -not -path "./.git/*" -exec sed -i \
  -e 's|/ssd/mkqin/miniconda3/etc/profile.d/conda.sh|/ssd/liuzirui/miniconda/etc/profile.d/conda.sh|g' \
  -e 's|conda activate vlabench_2|conda activate vlabench_openvla|g' \
  {} +
```

被改的 10 个文件：

- `VLABench/pipeline/nodes/node_logger.py`
- `VLABench/pipeline/nodes/simulation.py`
- `VLABench/pipeline/nodes/vlm_data.py`
- `VLABench/utils/verify_initial_state.py`
- `VLABench/utils/verify_lift_only.py`
- `VLABench/utils/verify_capopen_false.py`
- `VLABench/evaluation/model/policy/act.py`
- `scripts/convert_to_lerobot.py`
- `scripts/convert_to_lerobot_bak.py`
- `scripts/rename_task.py`

> 验证：`grep -rn "/ssd/mkqin" --include="*.py" .` 应返回空。

**注意**：`/ssd/liuzirui/lerobot` 这个目录**本机不存在**。`convert_to_lerobot.py` / `act.py` 改完后路径依然指向不存在的目录，运行这两文件仍会失败。本次任务（`test_e2e.py`）不依赖它们，所以不影响。

---

## 2. 安装 `langgraph`

```bash
/ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python -m pip install langgraph
```

会顺带装：langgraph 1.2.7、langgraph-checkpoint 4.1.1、langgraph-prebuilt 1.1.0、langgraph-sdk 0.4.2、langchain-core 1.4.8、langsmith 0.9.7、orjson 3.11.9、zstandard 0.25.0 等。无版本冲突。

---

## 3. 资产复用：软链接 `assets/obj`

`VLABench/assets/obj/` 目录未下载（README 提到要跑 `scripts/download_assets.py` 从 Google Drive 拉，~~本机连 GDrive 超时~~），而本机另一份原版 VLABench `/ssd/liuzirui/VLABench/VLABench/assets/obj/` 已有完整资产。

**做法**：在分支仓库里建软链接复用：

```bash
cd /ssd/liuzirui/VLAbench_LZR/VLABench/assets
ln -s /ssd/liuzirui/VLABench/VLABench/assets/obj obj
```

**注意**：往 `obj/` 路径下 `mkdir` 实际会写到**原版**那边（symlink 行为）。我尝试时建了 3 个空 lab_equipment 目录污染了原版，随后用 `rmdir` 撤销（仅空目录能删），原版已恢复干净。**以后别在 `obj/` 下用 `mkdir`**。

**实验器具资产（florence_flask / tripod / beaker / pipette / bunsen_burner / petri_dish / ...）**不在原版 `obj/` 里、但在**本分支自己的 `assets/review/`** 下都齐全——这正是本分支相比上游的增量。所以软链接 `obj/` 之后，跑 `place florence_flask on the tripod` 这类**实验室任务**所需的资产都能找到。

---

## 4. `constant.py` 加容错

`configs/constant.py` 在 import 时对 `name2class_xml` 里每一个 `obj/meshes/...` 路径做 `os.listdir`，**目录不存在**就 `FileNotFoundError`，整个 `import VLABench` 崩。原版少 3 个 lab_equipment 路径会触发这个。

**最小改动**：让 `get_object_list` 对缺失目录返回空列表（已有的实验器具在 `review/` 下，不走 `obj/`，不受影响）：

```python
# VLABench/configs/constant.py
def get_object_list(xml_dir, all=True, seen=True):
    ...
    xml_paths = []
    if not os.path.isdir(xml_dir):
        return []  # 资产目录缺失时容错，避免整个 import 崩
    subdirs = os.listdir(xml_dir)
    ...
```

`VLABench/assets/review/` 下的 30+ 种实验器具（alcohol_lamp、aspirin_pill_bottle、beaker_small/large、bunsen_burner、chemistry_lab_table、chemistry_tube、conical_flask、cylinder、drawer、drying_box、electronic_scale、flask、florence_flask、funnel、glass_stirring_rod、heat_device、hot_plate、lab_table、magnetic_stir_plate、mechanical_pipette、petri_dish、pill_bottle、pipette、pipettes_stand、rag、square_mat、thermometer、tripod、universal_support、water_bath）已全部就位。

---

## 5. 安装 `langchain_anthropic`（pipeline 真要调 LLM）

`pipeline/nodes/analyzer.py` 等节点会做：

```python
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None
```

缺包时 `analyzer_node` 走不到成功分支，`state["task_analysis"]` 不会被写回 state，后续 `normalizer_node` 就 KeyError。

`pipeline/config.py` 已经被前任写好了：

```python
MODEL_NAME: str = "claude-sonnet-4-6"
ANTHROPIC_API_KEY: str = "sk-bUVu80wi6nD0lMhBK7xYO2ZsJMFbHcZOGIOX9KuM8woHC3oM"
BASE_URL: str = "https://vip.aipro.love"   # 第三方代理
```

**做法**（在 `vlabench_openvla` env 里装包，不动 config.py）：

```bash
/ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python -m pip install langchain_anthropic
# 会顺带装：anthropic 0.116.0、langchain_anthropic 1.4.8、docstring_parser 0.18.0
```

**关键验证**（装完先 mock 调用一下，确认代理 / key 都能用，避免真跑任务才出错）：

```bash
/ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python -c "
import sys; sys.path.insert(0, '/ssd/liuzirui/VLAbench_LZR')
import os; os.environ['VLABENCH_ROOT'] = '/ssd/liuzirui/VLAbench_LZR/VLABench'
from VLABench.pipeline.config import AgentConfig
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(**AgentConfig.get_llm_config())
print('API OK ->', llm.invoke('回复: ok').content)
"
```

我们跑出 `API OK -> ok`——第三方代理 + key 都可用。

---

## 6. 装 ffmpeg（视频保存必需，且无 sudo 权限）

`simulation_node` 跑完后调 `mediapy.write_video` 存 mp4，里面走 `subprocess` 调 `ffmpeg`。本机：

- `sudo apt install ffmpeg` → `liuzirui is not in the sudoers file`
- `conda install -c conda-forge ffmpeg` → 报 `non-default solver backend (libmamba) but it was not recognized`
- `imageio-ffmpeg` 已经在 `vlabench_openvla` env 里了，**自带 ffmpeg 7.0.2 静态二进制**：

  ```
  /ssd/liuzirui/miniconda/envs/vlabench_openvla/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
  ```

**做法**：把它软链接到 `~/.local/bin/ffmpeg`（这个目录在 conda 自动配的 PATH 里）：

```bash
ln -sf /ssd/liuzirui/miniconda/envs/vlabench_openvla/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 \
       /ssd/liuzirui/.local/bin/ffmpeg
```

> ⚠️ 软链接指向具体 env 里的二进制。如果 `vlabench_openvla` env 被删 / `imageio-ffmpeg` 被重装，会失效，重建即可。

---

## 7. 端到端跑通结果

```bash
MUJOCO_GL=egl /ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python \
  /ssd/liuzirui/VLAbench_LZR/VLABench/pipeline/tests/test_e2e.py \
  "place florence_flask on the tripod"
```

输出关键行：

```
Registration: True                              # 7 节点 pipeline 成功
[Simulation] 开始仿真...
DEBUG [find_keypoint]: 成功找到有效抓取点
DEBUG [gently_pick]: ✓ 轻抓成功! stage_success=True
✓ 仿真成功!
  视频路径: /ssd/liuzirui/VLAbench_LZR/dataset/autogen_tasks/place_florence_flask/demo_0_success_True.mp4
```

视频元数据：

```
Duration: 00:00:12.00
Stream #0:0: Video: h264, yuv420p, 960x960, 10 fps, 306 kb/s
size: 451 KB
```

7 节点流水线（`analyzer → normalizer → asset_manager → skill/condition_planner → code_generator → registration → simulation`）全部跑通，物理仿真任务"place florence_flask on the tripod"成功完成。

---

## 8. 改的文件清单（git status）

代码改动（全部硬编码路径替换 + `constant.py` 1 行容错）：

```
M VLABench/configs/constant.py                       # 4. 加容错
M VLABench/evaluation/model/policy/act.py            # 1. 硬编码路径
M VLABench/pipeline/nodes/node_logger.py             # 1.
M VLABench/pipeline/nodes/simulation.py              # 1.
M VLABench/pipeline/nodes/vlm_data.py                # 1.
M VLABench/utils/verify_capopen_false.py             # 1.
M VLABench/utils/verify_initial_state.py             # 1.
M VLABench/utils/verify_lift_only.py                 # 1.
M scripts/convert_to_lerobot.py                      # 1.
M scripts/convert_to_lerobot_bak.py                  # 1.
M scripts/rename_task.py                             # 1.
```

非破坏性新建（git untracked）：

```
?? docs/SETUP_FIXES.md            # 本文档
?? ARCHITECTURE.md                # 项目架构
?? VLABench/assets/obj            # → /ssd/liuzirui/VLABench/VLABench/assets/obj 的符号链接
?? .local/bin/ffmpeg              # → imageio-ffmpeg 自带 ffmpeg 的符号链接
```

**额外发现的产物**（pipeline 跑成功后由 code_generator_node 生成，**这是 pipeline 工作流的成果**，不是 bug）：

```
M VLABench/tasks/autogen_tasks/place_florence_flask_series.py   # 58 行
```

内容是一个 `PlaceFlorenceFlaskConfigManager`（继承 `BenchTaskConfigManager`），用 `name2class_xml["florence_flask"]` / `name2class_xml["tripod"]` 的资产路径，把 `florence_flask_0` 随机放在 `[0.05, 0.15] × [-0.15, -0.05]` 范围、`tripod_0` 放在 `[0.35, 0.45] × [-0.05, 0.05]` 范围。pipeline 跑完一个新指令就会在 `tasks/autogen_tasks/` 下生成对应的 `*_series.py`。

未跟踪（用户原有）：

```
?? "任务表格_类别1_液体转移与分液.xlsx"
?? "给子睿的工作指南.md"
```

环境外改动（`~/.bashrc`，不在这台机器的 git 仓库里）：

```diff
+ conda activate vlabench_openvla 2>/dev/null || true   # 见第 9 节
```

---

## 9. 长期：让每个新 shell 默认就是 vlabench_openvla

**问题**：`conda 26.3.2` 在新 shell 里要求 `conda init`，且默认 env 是 base。每次开新终端：
- `conda activate vlabench_openvla` 报 `CondaError: Run 'conda init' before 'conda activate'`，PATH 不变
- 但提示符看起来"激活了"——是 Anaconda Prompt 主题的视觉欺骗，**`which python` 还是 base 的 Python 3.13**
- 直接 `python` 跑就 `ModuleNotFoundError: dm_control`

**一次性修好**：在 `~/.bashrc` 末尾追加：

```bash
# 自动激活 vlabench_openvla（默认工作环境）
# 如需切回 base，请执行 `conda deactivate`
conda activate vlabench_openvla 2>/dev/null || true
```

（`|| true` 是为了在 `source conda.sh` 还没执行时不报错，conda init 那块会先 source 它。）

**效果**：新 shell / VSCode 终端 / `bash -i` 启动后自动激活，`which python` 直接是 `vlabench_openvla/bin/python`（Python 3.10），`dm_control` 立即可用。

> 注意：自动激活只发生在**交互式 shell**（`bash -i`）。`#!/bin/bash` 脚本里不会自动激活，脚本里要么 `source ... && conda activate ...`，要么直接用绝对路径 `/ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python`。

**手动验证**（模拟新 shell）：

```bash
bash -i -c 'which python; python -c "import dm_control; print(\"dm_control OK\")"'
# 输出: /ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python
#       dm_control OK
```

---

## 10. 怎么验证当前状态

```bash
# 1) 验证 import 链全部就绪
/ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python -c "
import sys; sys.path.insert(0, '/ssd/liuzirui/VLAbench_LZR')
import os
os.environ['VLABENCH_ROOT'] = '/ssd/liuzirui/VLAbench_LZR/VLABench'
os.environ['MUJOCO_GL'] = 'egl'
import VLABench.tasks.components
from VLABench.pipeline.agent import build_vlabench_agent, create_initial_state
from VLABench.pipeline.nodes.analyzer import analyzer_node
from VLABench.pipeline.nodes.normalizer import normalizer_node
from VLABench.pipeline.nodes.asset_manager import asset_manager_node
from VLABench.pipeline.nodes.skill_planner import skill_planner_node
from VLABench.pipeline.nodes.condition_planner import condition_planner_node
from VLABench.pipeline.nodes.code_generator import code_generator_node
from VLABench.pipeline.nodes.registration import registration_node
from VLABench.pipeline.nodes.simulation import simulation_node
from VLABench.pipeline.nodes.node_logger import init_run_log, log_instruction_entry
print('ALL IMPORTS OK')
"
# 应输出: ALL IMPORTS OK

# 2) 验证 ffmpeg 可用
which ffmpeg   # /ssd/liuzirui/.local/bin/ffmpeg
ffmpeg -version | head -1   # ffmpeg version 7.0.2-static ...

# 3) 跑完整 pipeline + 仿真
MUJOCO_GL=egl /ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python \
  /ssd/liuzirui/VLAbench_LZR/VLABench/pipeline/tests/test_e2e.py \
  "place florence_flask on the tripod"
# 应输出: Registration: True  +  ✓ 仿真成功!  + 视频路径
```

---

## 11. 后续可能想做的

1. **测试其他指令**：`"lift the beaker"`、`"pick up the pipette"`、`"unscrew the bottle"` 等，看 pipeline 对不同指令的泛化能力。
2. **看仿真视频**：把生成的 mp4 拷到本机看，或在服务器上 `mpv <video_path>` 播放（要 X11 forward）。
3. **生成新任务**：pipeline 跑成功后会在 `tasks/autogen_tasks/` 下生成 `place_florence_flask_*.py` 系列文件，可作为模板派生更多化学实验任务。
4. **长期方案**：
   - 装真正的 conda ffmpeg 包：先 `conda install -n base conda-libmamba-solver`，再 `conda install -n vlabench_openvla -c conda-forge ffmpeg`
   - 评估 `imageio-ffmpeg` 升级是否会更稳
   - 把 `pipeline/config.py` 里写死的 key 改成读环境变量（避免进 git 历史）

---

## 撤回指南

如需重置环境到本次修改之前：

```bash
cd /ssd/liuzirui/VLAbench_LZR
git checkout -- VLABench/ scripts/                # 撤回所有代码改动
rm -f docs/SETUP_FIXES.md ARCHITECTURE.md          # 删新增文档
unlink VLABench/assets/obj                         # 撤回资产软链接
unlink /ssd/liuzirui/.local/bin/ffmpeg             # 撤回 ffmpeg 软链接

# 撤回 .bashrc 自动激活（可选）
sed -i '/^# 自动激活 vlabench_openvla/,$d' ~/.bashrc
```

`/ssd/liuzirui/miniconda/envs/vlabench_openvla` 里的 `langgraph / langchain_anthropic / anthropic` 包是装在 env 里的、不是 git 跟踪范围，撤回时不会影响；如需连包一起卸：`/ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/pip uninstall langgraph langchain_anthropic anthropic -y`。
