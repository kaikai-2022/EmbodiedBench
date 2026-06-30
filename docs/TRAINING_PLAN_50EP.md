# 训练计划：ACT 多任务 50 Epochs 多卡训练

## 数据集信息

- **数据集**: vlabench_lift_flask_place_beaker_200ep
- **总帧数**: 16,323 帧
- **Episodes**: 200 (lift_flask 100 + place_beaker 100)
- **Features**: observation.image, observation.wrist_image, observation.state (7D), action (7D)

## 训练配置

### 硬件配置
- **GPU**: 4 × RTX 4090 (DDP)
- **batch_size**: 2 per GPU → 全局 batch = 8
- **num_workers**: 2 per GPU

### 训练步数计算

```
steps_per_epoch = total_frames / global_batch_size
                = 16,323 / 8
                = ~2,040 步

total_steps = 50 epochs × 2,040 steps/epoch
            = ~102,000 步
```

**训练时长预估**:
- 从之前日志: 5 epochs (10,000 DDP步) = 4.5 小时
- 50 epochs 预估 = 4.5 × 10 = **45 小时**
- 考虑 checkpoint/评估开销: **~50 小时**

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--offline.steps` | 102000 | 50 epochs |
| `--batch_size` | 2 | per GPU |
| `--num_workers` | 2 | per GPU (从0提升，避免数据瓶颈) |
| `--save_freq` | 10000 | 每 5 epochs 保存一次 |
| `--eval_freq` | 2000 | 每 1 epoch 评估一次 |
| `--log_freq` | 100 | 每 100 步记录日志 |
| `--learning_rate` | 1e-5 | 保持不变 |
| `--seed` | 1000 | 可复现 |

### Checkpoint 保存计划

| Checkpoint | Step | Epoch | 预估时间 |
|------------|------|-------|---------|
| 001000 | 1,000 | 0.5 | ~0.5h |
| 002000 | 2,000 | 1 | ~1h |
| ... | ... | ... | ... |
| 010000 | 10,000 | 5 | ~4.5h |
| 020000 | 20,000 | 10 | ~9h |
| ... | ... | ... | ... |
| 100000 | 100,000 | 49 | ~45h |
| 102000 | 102,000 | 50 | ~46h |

总计 **11 个 checkpoint** (每 10,000 步一个)

## 启动命令

```bash
cd /ssd/mkqin/workspace/lerobot

OUTPUT_DIR="/ssd/mkqin/workspace/VLABench/models/act_lift_flask_place_beaker_200ep_50ep_$(date +%Y%m%d_%H%M%S)"

torchrun --nproc_per_node=4 --master_port=29500 \
    lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id="vlabench_lift_flask_place_beaker_200ep" \
    --dataset.local_files_only=true \
    --dataset.image_transforms.enable=false \
    --dataset.use_imagenet_stats=true \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=2 \
    --num_workers=2 \
    --offline.steps=102000 \
    --save_freq=10000 \
    --eval_freq=2000 \
    --log_freq=100 \
    --device=cuda \
    --seed=1000 \
    > "$OUTPUT_DIR/train.log" 2>&1
```

**注意事项**:
1. 不预先创建 `$OUTPUT_DIR`（lerobot 会自动创建）
2. 使用后台运行 + 日志重定向
3. 每 5 epochs 保存 checkpoint，防止意外中断丢失过多进度

## 验证目标

训练完成后应达到：

| 指标 | 目标 |
|------|------|
| Loss | < 1.0 (收敛) |
| 图像敏感度 | > 0.01 (学会用图像) |
| Chunk 内部变化 | > 0.05 (学会时间序列) |
| 推理成功率 | > 30% (至少能抓起物体) |

## 监控计划

训练期间监控：

```bash
# 查看最新 loss
tail -f "$OUTPUT_DIR/train.log" | grep "loss"

# 检查 GPU 使用
watch -n 5 nvidia-smi

# 检查 checkpoint 进度
ls -lth "$OUTPUT_DIR/checkpoints/"
```

每 10 小时检查一次：
1. Loss 是否持续下降
2. 是否有 NaN/Inf
3. checkpoint 是否正常保存