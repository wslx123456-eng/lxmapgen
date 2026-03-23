# Ubuntu Real-Data Launch Checklist

## 1. 这份清单解决什么问题

这份清单是给 Ubuntu 服务器直接启动真实数据训练/推理用的。

目标是三件事：

- 快速装好环境
- 快速挂上真实数据和本地权重
- 快速启动训练、继续训练、推理和评估

## 2. 当前代码状态

这套代码已经在本机做过一次真实执行层面的 smoke 验证，验证内容是：

- `check_geo_data`
- `train_geo_full`
- `predict_geo_vector`
- `eval_geo_vector`

但那次验证使用的是：

- tiny GPT2
- tiny ViT
- 合成 GeoTIFF / GeoJSON 数据

所以可以确认“工程主链路能跑”，但不代表真实数据效果已经验证。

## 2.1 Win Smoke 不等于 Ubuntu 已确认

不能把“Windows 上 smoke 跑通”直接等同于“Ubuntu 上一定跑通”。

Windows smoke 目前只能证明：

- 代码主链路没有明显的工程级断点
- `check -> train -> predict -> eval` 这条链路已经真实执行过
- 当前 `geo` 代码没有依赖 PowerShell 才能运行

但 Ubuntu 侧仍然有 4 类独立风险需要单独确认：

1. `torch` 和服务器 CUDA 版本是否匹配
2. `rasterio / pyproj` 在该 Ubuntu 上是否缺系统库
3. `DINOV2_BACKBONE_PATH / QWEN_MODEL_PATH` 是否真的是 Linux 可读的本地权重目录
4. 当前配置里的 `local_files_only: true` 是否符合你的服务器使用方式

我已经做过一轮静态跨平台检查，当前结论是：

- `geo` 主链路运行代码没有写死 Windows 路径
- 运行代码主要使用 `os.path` / `Path`，路径拼接本身是跨平台的
- Ubuntu 入口脚本 `scripts/*.sh` 使用的是 LF 换行，不是 CRLF
- 训练/推理/评估主入口没有依赖 PowerShell 命令

所以更准确的结论应该是：

- 当前代码“具备 Ubuntu 运行条件”
- 但“是否真正跑通”仍然要以 Ubuntu 上第一次实跑为准

## 3. Ubuntu 推荐环境

建议：

- Ubuntu 20.04 / 22.04 / 24.04
- `conda` 或 `miniconda`
- Python `3.10`
- GPU 训练时使用与你机器 CUDA 匹配的 PyTorch

当前本地实际装过并能导入的包版本大致是：

- `python 3.10`
- `torch 2.10.0`
- `transformers 5.3.0`
- `accelerate 1.13.0`
- `peft 0.18.1`
- `rasterio 1.4.4`
- `pyproj 3.7.1`
- `shapely 2.1.2`

Ubuntu 上不需要强行锁死这些版本，但建议从这组开始。

## 4. 一次性安装命令

在仓库根目录执行：

```bash
cd /path/to/UniMapGenStrongBaseline

conda create -n unimapgen-geo python=3.10 pip -y
conda activate unimapgen-geo
```

### 4.1 先装 PyTorch

GPU 机器优先按官方命令安装匹配 CUDA 的 PyTorch。

例如，如果你的服务器是 CUDA 12.1，可以先用类似：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你只想先做 CPU 自检：

```bash
pip install torch torchvision torchaudio
```

### 4.2 再装项目依赖

```bash
pip install -r requirements-geo.txt
```

### 4.3 可选自检

```bash
python -c "import torch, transformers, peft; print('model deps ok')"
python -c "import rasterio, pyproj, shapely; print('geo deps ok')"
```

如果 `rasterio` / `pyproj` 在你的 Ubuntu 上安装失败，再补系统库：

```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev proj-bin libproj-dev
pip install -r requirements-geo.txt
```

## 5. 真实数据和权重路径

代码默认从配置里读三类路径：

- 数据集根目录
- DINOv2 本地权重目录
- Qwen 本地权重目录

最简单的用法是不改 YAML，直接用环境变量覆盖。

### 5.1 你需要准备的目录

假设：

- 真实数据在 `/data/geo_vector_dataset`
- DINOv2 在 `/data/ckpts/dinov2`
- Qwen 在 `/data/ckpts/qwen`

那就在 shell 里执行：

```bash
export UNIMAPGEN_GEO_ROOT=/data/geo_vector_dataset
export DINOV2_BACKBONE_PATH=/data/ckpts/dinov2
export QWEN_MODEL_PATH=/data/ckpts/qwen
```

### 5.2 权重目录要求

`DINOV2_BACKBONE_PATH` 和 `QWEN_MODEL_PATH` 当前默认按“本地 HF snapshot”解析，目录需要满足以下任一形式：

1. 目录下直接有 `config.json`
2. 目录是 Hugging Face cache 风格，里面有：
   - `refs/main`
   - `snapshots/<hash>/config.json`

如果你希望在线从 Hugging Face 下载，而不是只读本地目录，需要把配置里的：

```yaml
model:
  local_files_only: false
```

## 6. 真实数据目录结构

当前代码默认的数据结构是：

```text
dataset_root/
  train/
    sample_xxx/
      patch_tif/
        0.tif
        0_edit_poly.tif
      label_check_crop/
        Lane.geojson
        Intersection.geojson
  val/
    ...
  test/
    ...
```

完整说明见：

- [Geo数据集结构说明](/c:/DevelopProject/VScode/geomapgen/docs/Geo数据集结构说明.md)

## 7. 第一次启动前的最短检查

先做数据自检，不要直接开训练。

### LoRA 配置自检

```bash
python -m unimapgen.check_geo_data --config configs/geo_vector_lora.yaml
```

### 全参配置自检

```bash
python -m unimapgen.check_geo_data --config configs/geo_vector_full.yaml
```

如果这里报错，优先看错误码文档：

- 错误码文档已移除，如需排查请直接查看训练/推理日志中的 `GEO-xxxx` 编号。

## 8. 快速启动训练

### 8.1 LoRA 训练

直接跑模块：

```bash
python -m unimapgen.train_geo_lora --config configs/geo_vector_lora.yaml
```

或者用封装脚本：

```bash
bash scripts/run_geo_lora_train.sh configs/geo_vector_lora.yaml
```

### 8.2 全量训练

直接跑模块：

```bash
python -m unimapgen.train_geo_full --config configs/geo_vector_full.yaml
```

或者用封装脚本：

```bash
bash scripts/run_geo_full_train.sh configs/geo_vector_full.yaml
```

### 8.3 输出目录规则

第一次训练默认会在：

- `outputs/geo_vector_lora/<timestamp>/`
- `outputs/geo_vector_full/<timestamp>/`

生成：

- `config_snapshot.yaml`
- `run_meta.txt`
- `metrics.jsonl`
- `latest.pt`
- `best.pt`

## 9. 继续训练

继续训练时，不建议直接改原始主配置。更稳的方式是另存一份 resume 配置，例如：

- `configs/geo_vector_lora_resume.yaml`
- `configs/geo_vector_full_resume.yaml`

至少修改下面这几项：

```yaml
train:
  init_checkpoint: /path/to/your/previous_run/latest.pt
  resume_in_place: true
  resume_optimizer: true
  resume_scaler: true
```

然后继续训练：

```bash
python -m unimapgen.train_geo_lora --config configs/geo_vector_lora_resume.yaml
```

或：

```bash
python -m unimapgen.train_geo_full --config configs/geo_vector_full_resume.yaml
```

当前逻辑是：

- 首次训练：新建时间戳目录
- `resume_in_place=true` 时：继续写回 checkpoint 所在目录

## 10. 快速启动推理

推理支持三种模式：

- 单张 tif
- 整个目录
- 按 split 扫描数据集

### 10.1 单张 tif 推理

```bash
python -m unimapgen.predict_geo_vector \
  --config configs/geo_vector_lora.yaml \
  --checkpoint outputs/geo_vector_lora/20260311_120000/latest.pt \
  --input_image /data/infer/sample_001.tif \
  --output_dir outputs/geo_predictions_single
```

### 10.2 批量目录推理

```bash
python -m unimapgen.predict_geo_vector \
  --config configs/geo_vector_lora.yaml \
  --checkpoint outputs/geo_vector_lora/20260311_120000/latest.pt \
  --input_dir /data/infer_tif \
  --glob '*.tif' \
  --output_dir outputs/geo_predictions_dir
```

### 10.3 按 split 推理

```bash
python -m unimapgen.predict_geo_vector \
  --config configs/geo_vector_lora.yaml \
  --checkpoint outputs/geo_vector_lora/20260311_120000/latest.pt \
  --split test \
  --output_dir outputs/geo_predictions_test
```

### 10.4 用脚本推理

单图：

```bash
bash scripts/run_geo_predict.sh \
  configs/geo_vector_lora.yaml \
  outputs/geo_vector_lora/20260311_120000/latest.pt \
  image \
  /data/infer/sample_001.tif \
  outputs/geo_predictions_single
```

目录：

```bash
bash scripts/run_geo_predict.sh \
  configs/geo_vector_lora.yaml \
  outputs/geo_vector_lora/20260311_120000/latest.pt \
  dir \
  /data/infer_tif \
  outputs/geo_predictions_dir
```

split：

```bash
bash scripts/run_geo_predict.sh \
  configs/geo_vector_lora.yaml \
  outputs/geo_vector_lora/20260311_120000/latest.pt \
  split \
  test \
  outputs/geo_predictions_test
```

### 10.5 推理输出

每个样本输出一个目录，里面至少会有：

- `Lane.geojson`
- `Intersection.geojson`

并带一份：

- `summary.json`

## 11. 快速启动评估

### 11.1 评估验证集

```bash
python -m unimapgen.eval_geo_vector \
  --config configs/geo_vector_lora.yaml \
  --checkpoint outputs/geo_vector_lora/20260311_120000/latest.pt \
  --split val \
  --output outputs/geo_eval_metrics_val.json
```

### 11.2 评估测试集

```bash
python -m unimapgen.eval_geo_vector \
  --config configs/geo_vector_lora.yaml \
  --checkpoint outputs/geo_vector_lora/20260311_120000/latest.pt \
  --split test \
  --output outputs/geo_eval_metrics_test.json
```

### 11.3 用脚本评估

```bash
bash scripts/run_geo_eval.sh \
  configs/geo_vector_lora.yaml \
  outputs/geo_vector_lora/20260311_120000/latest.pt \
  val \
  outputs/geo_eval_metrics_val.json
```

## 12. Ubuntu 上最常改的配置项

第一次上真实数据，优先看这几项：

- `data.dataset_root`
- `model.dino_model_path`
- `model.qwen_model_path`
- `train.batch_size`
- `train.val_batch_size`
- `train.amp`
- `tiling.train.tile_size_px`
- `tiling.predict.tile_size_px`
- `tiling.train.overlap_px`
- `postprocess.line_dedup_distance_m`
- `postprocess.polygon_dedup_iou`

如果 GPU 显存不够，优先这样调：

1. 减小 `train.batch_size`
2. 开启 `train.amp: true`
3. 减小 `data.image_size`
4. 减小 `tiling.train.tile_size_px`

## 13. 真实启动的建议顺序

推荐严格按下面顺序来：

1. 安装环境
2. 导出 3 个路径环境变量
3. 跑 `check_geo_data`
4. 跑 LoRA 训练
5. 产出 checkpoint 后跑一次单图推理
6. 再跑 `eval_geo_vector`
7. 确认输出无误后再开长训

## 14. 最短命令版

如果你只想保留最短的启动片段，下面这组命令就够：

```bash
cd /path/to/UniMapGenStrongBaseline

conda create -n unimapgen-geo python=3.10 pip -y
conda activate unimapgen-geo

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-geo.txt

export UNIMAPGEN_GEO_ROOT=/data/geo_vector_dataset
export DINOV2_BACKBONE_PATH=/data/ckpts/dinov2
export QWEN_MODEL_PATH=/data/ckpts/qwen

python -m unimapgen.check_geo_data --config configs/geo_vector_lora.yaml
python -m unimapgen.train_geo_lora --config configs/geo_vector_lora.yaml
```

训练完后最短推理命令：

```bash
python -m unimapgen.predict_geo_vector \
  --config configs/geo_vector_lora.yaml \
  --checkpoint outputs/geo_vector_lora/<timestamp>/latest.pt \
  --split test \
  --output_dir outputs/geo_predictions_test
```

## 15. 如果出错，优先给我什么

只给我这三项就够：

1. 错误码，比如 `GEO-1204`
2. 报错整行
3. 你执行的命令

错误码对照见：

- 错误码文档已移除，如需排查请直接查看训练/推理日志中的 `GEO-xxxx` 编号。
