# OpenPI Policy Evaluation Guide

## Overview

This guide covers how to run VLABench evaluation with OpenPI (Physical Intelligence) policy. The evaluation requires two terminals:
- **Terminal 1**: Run the OpenPI policy server (provides inference service)
- **Terminal 2**: Run the VLABench evaluation client (connects to server)

## Prerequisites

### 1. Environment Setup

You need two separate environments:

#### Environment 1: OpenPI Virtual Environment (.venv)
Located at: `third_party/openpi/examples/vlabench/.venv`

This environment contains OpenPI and all its dependencies including JAX for model inference.

```bash
# The .venv should already be set up. If not, create it:
cd /ssd/mkqin/workspace/VLABench/third_party/openpi
/ssd/mkqin/miniconda3/bin/uv venv --python 3.11 examples/vlabench/.venv
source examples/vlabench/.venv/bin/activate
/ssd/mkqin/miniconda3/bin/uv pip sync examples/vlabench/requirements.txt
/ssd/mkqin/miniconda3/bin/uv pip install -e packages/openpi-client
/ssd/mkqin/miniconda3/bin/uv pip install -e /ssd/mkqin/workspace/VLABench
```

#### Environment 2: VLABench Conda Environment
Use the existing `vlabench_2` conda environment:

```bash
conda activate vlabench_2
pip install openpi-client
```

### 2. Required Files

#### Checkpoint Files
The model checkpoint should be downloaded to:
- `/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task/`

If not present, the server will automatically download from S3 on first run.

#### Tokenizer File
Required for PaliGemma tokenizer:
- `/ssd/mkqin/.cache/openpi/big_vision/paligemma_tokenizer.model`

If you encounter network issues downloading from Google Cloud Storage, download from HuggingFace:
```bash
# Download from HuggingFace mirror
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/paligemma-3b-pt-224', 'tokenizer.model', local_dir='/tmp/tokenizer')"
cp /tmp/tokenizer/tokenizer.model /ssd/mkqin/.cache/openpi/big_vision/paligemma_tokenizer.model
mkdir -p /ssd/mkqin/.cache/openpi/big_vision
```

## Running Evaluation

### Terminal 1: Start the Policy Server

```bash
# 1. Activate the OpenPI virtual environment
source /ssd/mkqin/workspace/VLABench/third_party/openpi/examples/vlabench/.venv/bin/activate

# 2. Set environment variables
export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl

# 3. Select a GPU (check with nvidia-smi, avoid GPU 0 if it's nearly full)
export CUDA_VISIBLE_DEVICES=1

# 4. Go to OpenPI directory
cd /ssd/mkqin/workspace/VLABench/third_party/openpi

# 5. Start the server
python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pifast_ft_vlabench_primitive --policy.dir=/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task
```

**Expected output:**
```
INFO:root:Loading model...
INFO:absl:Finished restoring checkpoint from /ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task/params.
INFO:root:Creating server (host: ubuntu, ip: 127.0.1.1)
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### Terminal 2: Run the Evaluation Client

Wait until the server is fully started (you see "server listening on 0.0.0.0:8000"), then:

```bash
# 1. Activate VLABench conda environment
source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2

# 2. Set environment variables
export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl

# 3. Go to OpenPI directory
cd /ssd/mkqin/workspace/VLABench/third_party/openpi

# 4. Run evaluation with specific tasks
python examples/vlabench/eval.py \
    --args.host localhost \
    --args.port 8000 \
    --args.tasks "select_toy" \
    --args.n_episode 1 \
    --args.save_dir /ssd/mkqin/workspace/VLABench/logs/openpi_eval

# Or run with evaluation track
python examples/vlabench/eval.py \
    --args.host localhost \
    --args.port 8000 \
    --args.eval_track track_1_in_distribution \
    --args.n_episode 10 \
    --args.save_dir /ssd/mkqin/workspace/VLABench/logs/openpi_eval
```

## Command Line Arguments

### Server Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--port` | Server port | 8000 |
| `policy:checkpoint` | Use checkpoint instead of default policy | - |
| `--policy.config` | Config name for the policy | pifast_ft_vlabench_primitive |
| `--policy.dir` | Checkpoint directory path | - |

### Client Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--args.host` | Server host | localhost |
| `--args.port` | Server port | 8000 |
| `--args.tasks` | Space-separated task names | None |
| `--args.eval_track` | Evaluation track name | None |
| `--args.n_episode` | Number of episodes per task | 50 |
| `--args.save_dir` | Directory to save results | Required |
| `--args.replan_steps` | Steps before replanning | 5 |
| `--args.visulization` | Enable visualization | True |

## Available Evaluation Tracks

| Track | Description |
|-------|-------------|
| `track_1_in_distribution` | Domain-in distribution evaluation |
| `track_2_cross_category` | Cross-category generalization |
| `track_3_common_sense` | Common sense understanding |
| `track_4_semantic_instruction` | Semantic instruction understanding |
| `track_6_unseen_texture` | Unseen texture robustness |

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'flax' or 'openpi'

**Problem:** The OpenPI package is not installed in the virtual environment.

**Solution:**
```bash
source /ssd/mkqin/workspace/VLABench/third_party/openpi/examples/vlabench/.venv/bin/activate
cd /ssd/mkqin/workspace/VLABench/third_party/openpi
/ssd/mkqin/miniconda3/bin/uv pip install -e . --no-deps
```

### 2. ModuleNotFoundError: No module named 'openpi_client'

**Problem:** The client environment doesn't have openpi-client installed.

**Solution:**
```bash
conda activate vlabench_2
pip install openpi-client
```

### 3. Segfault or AttributeError in MuJoCo/OpenGL

**Problem:** Rendering issues on headless server.

**Solution:** Set the following environment variables:
```bash
export MUJOCO_GL=egl
```

### 4. jaxlib.xla_extension.XlaRuntimeError: INTERNAL: Failed to initialize BLAS support

**Problem:** GPU memory is full or wrong GPU is selected.

**Solution:** Check GPU usage and select an available GPU:
```bash
nvidia-smi  # Check GPU memory usage
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1, 2, etc.
```

### 5. Network issues when downloading checkpoints

**Problem:** Cannot connect to S3 or HuggingFace.

**Solution:** Use HuggingFace mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 6. ModuleNotFoundError: No module named 'rrt_algorithms'

**Problem:** VLABench dependencies are incomplete.

**Solution:**
```bash
source /ssd/mkqin/workspace/VLABench/third_party/openpi/examples/vlabench/.venv/bin/activate
/ssd/mkqin/miniconda3/bin/uv pip install -e /ssd/mkqin/workspace/VLABench/src/rrt-algorithms --no-deps
```

### 7. "Failed to converge after 99 steps" warnings

**Problem:** JAX optimizer warnings during model inference.

**Solution:** These are normal warnings and do not affect evaluation. The evaluation will continue normally.

### 8. UnboundLocalError: cannot access local variable 'tasks'

**Problem:** Neither `--args.tasks` nor `--args.eval_track` was specified.

**Solution:** Always specify at least one of these arguments:
```bash
--args.tasks "select_toy"
# OR
--args.eval_track track_1_in_distribution
```

## GPU Selection

Check GPU availability before starting the server:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Select a GPU with sufficient free memory (at least 8GB recommended for model inference):
```bash
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1
```

## Results

Evaluation results are saved to the specified `--args.save_dir`:

```
logs/
├── metrics.json              # Summary metrics
└── task_name/
    ├── detail_info.json      # Detailed episode info
    └── videos/               # Visualization videos (if enabled)
```

Example metrics.json:
```json
{
    "select_toy": {
        "success_rate": 0.0,
        "intention_score": 0.0,
        "progress_score": 0.5
    }
}
```

## Quick Start Script

Save the following as `run_openpi_eval.sh` for convenience:

```bash
#!/bin/bash

# Configuration
PORT=8000
GPU_ID=1
CHECKPOINT_DIR=/ssd/mkqin/.cache/openpi/vlabench_checkpoints/pi0_fast_primitive_10task
SAVE_DIR=/ssd/mkqin/workspace/VLABench/logs/openpi_eval
TASK="select_toy"
N_EPISODE=1

# Terminal 1: Start server (run this first)
start_server() {
    source /ssd/mkqin/workspace/VLABench/third_party/openpi/examples/vlabench/.venv/bin/activate
    export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
    export HF_ENDPOINT=https://hf-mirror.com
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    cd /ssd/mkqin/workspace/VLABench/third_party/openpi
    python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=pifast_ft_vlabench_primitive --policy.dir=$CHECKPOINT_DIR
}

# Terminal 2: Run evaluation
run_eval() {
    source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
    conda activate vlabench_2
    export PYTHONPATH=/ssd/mkqin/workspace/VLABench/third_party/openpi/src:/ssd/mkqin/workspace/VLABench:$PYTHONPATH
    export HF_ENDPOINT=https://hf-mirror.com
    export MUJOCO_GL=egl
    cd /ssd/mkqin/workspace/VLABench/third_party/openpi
    python examples/vlabench/eval.py --args.host localhost --args.port $PORT --args.tasks "$TASK" --args.n_episode $N_EPISODE --args.save_dir $SAVE_DIR
}

case "$1" in
    server) start_server ;;
    eval) run_eval ;;
    *) echo "Usage: $0 {server|eval}" ;;
esac
```

Usage:
```bash
chmod +x run_openpi_eval.sh

# Terminal 1: Start server
./run_openpi_eval.sh server

# Terminal 2: Run evaluation (after server starts)
./run_openpi_eval.sh eval
```

## Citation

If you use OpenPI in your research, please cite:
```bibtex
@article{openpi,
  title={OpenPi},
  author={Physical Intelligence},
  year={2024},
  url={https://github.com/Physical-Intelligence/openpi}
}
```
