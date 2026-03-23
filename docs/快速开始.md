# Quickstart

## 1. 项目定位

这个副本项目只服务于一件事：

- 复现当前 strongest baseline：`baseline + semantic init`

默认主配置：

- 从头训练：`configs/strongest_baseline_train.yaml`
- 复现 strongest resume：`configs/strongest_baseline_resume.yaml`
- 最小可运行检查：`configs/strongest_baseline_smoke.yaml`

## 2. 数据与权重默认路径

默认直接复用当前机器上已经准备好的数据和权重：

- AV2/OpenSatMap 数据：
  - `/mnt/data/project/jn/UniMapGen/data_samples/av2_opensatmap_partial_fix`
- DINOv2：
  - `/mnt/data/project/jn/UniMapGen/ckpts/dinov2_vitl14/models--facebook--dinov2-large`
- Qwen2.5-1.5B：
  - `/mnt/data/project/jn/UniMapGen/ckpts/qwen2.5/models--Qwen--Qwen2.5-1.5B`

如果路径不同，可以通过环境变量覆盖：

- `AV2_ALIGNED_PARTIAL_ROOT`
- `AV2_ALIGNED_PARTIAL_ANN_JSON`
- `AV2_ALIGNED_PARTIAL_SPLIT_DIR`
- `DINOV2_BACKBONE_PATH`
- `QWEN_MODEL_PATH`
- `STRONGEST_BASE_INIT_CKPT`

## 3. 模型结构

训练链路很简单：

1. 读 satellite patch 图像
2. 用 DINOv2 提取 satellite tokens
3. 用文本 prompt 作为 instruction prefix
4. 用 Qwen 自回归生成序列化 map tokens
5. 用 semantic init 初始化新增 map token embedding

核心文件：

- 数据集：`unimapgen/data/qwen_map_dataset.py`
- map tokenizer：`unimapgen/data/serialization.py`
- Qwen tokenizer 扩展：`unimapgen/data/qwen_map_tokenizer.py`
- 模型：`unimapgen/models/qwen_map_generator.py`
- pipeline：`unimapgen/qwen_map_pipeline.py`
- 训练：`unimapgen/train_qwen_map.py`

## 4. 常用命令

先做 smoke：

```bash
cd /mnt/data/project/zsy/UniMapGenStrongBaseline
bash scripts/run_smoke.sh
```

从头训练 strongest baseline：

```bash
cd /mnt/data/project/zsy/UniMapGenStrongBaseline
bash scripts/run_train.sh
```

按当前 strongest 的方式做 resume：

```bash
cd /mnt/data/project/zsy/UniMapGenStrongBaseline
bash scripts/run_resume_strongest.sh
```

评估 resume 结果：

```bash
cd /mnt/data/project/zsy/UniMapGenStrongBaseline
bash scripts/eval_resume_strongest.sh
```

## 5. 输出文件

从头训练输出：

- `outputs/strongest_baseline_train/metrics.jsonl`
- `outputs/strongest_baseline_train/best.pt`

resume strongest 输出：

- `outputs/strongest_baseline_resume/metrics.jsonl`
- `outputs/strongest_baseline_resume/best.pt`
- `outputs/strongest_baseline_resume/official_metrics_val.json`

## 6. 给接手同学的建议

先只做这三件事：

1. 跑 `run_smoke.sh`
2. 确认 `run_train.sh` 能启动
3. 再做 `run_resume_strongest.sh` 和 `eval_resume_strongest.sh`

不要在这个副本项目里直接加回 PV/state 等支线，除非先明确它们是必须的。
