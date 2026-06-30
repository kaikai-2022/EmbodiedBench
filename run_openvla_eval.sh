#!/bin/bash
set -e

export VLABENCH_ROOT=/ssd/mkqin/workspace/VLABench/VLABench
export PYTHONPATH=/ssd/mkqin/workspace/VLABench
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_openvla

MODEL_CKPT=/ssd/mkqin/workspace/VLABench/models/openvla-7b-vlabench-primitive-lora
# VLABench/openvla-lora is a full merged model, no PEFT adapter needed
# pass --lora_ckpt None or omit so openvla.py skips PeftModel.from_pretrained
SAVE_DIR=/ssd/mkqin/workspace/VLABench/openvla_eval_results

cd /ssd/mkqin/workspace/VLABench

python scripts/evaluate_policy.py \
    --policy openvla \
    --model_ckpt "$MODEL_CKPT" \
    --lora_ckpt None \
    --tasks select_fruit \
    --n-episode 1 \
    --save-dir "$SAVE_DIR" \
    --metrics success_rate intention_score progress_score \
    --visulization