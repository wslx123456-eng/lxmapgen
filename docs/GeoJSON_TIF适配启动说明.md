# v1.0 GeoJSON + tif 适配启动说明

## 1. 这套适配器做了什么

这次加的是一个薄适配层，用来把当前项目的数据：

- `patch_tif/0.tif`
- `label_check_crop/Lane.geojson`
- `label_check_crop/Intersection.geojson`

导出成 `unimapgen-v1.0` 现成的 `LLaMAFactory ShareGPT + images` 数据格式。

当前适配器脚本：

- [export_llamafactory_from_geo_patch_dataset.py](/c:/DevelopProject/VScode/geomapgen/unimapgen-v1.0/scripts/export_llamafactory_from_geo_patch_dataset.py)
- [run_geo_patch_v1_lf_train.sh](/c:/DevelopProject/VScode/geomapgen/unimapgen-v1.0/scripts/run_geo_patch_v1_lf_train.sh)

## 2. 当前版本的边界

当前适配器只导出：

- `Lane.geojson`

并把它转换成 `v1.0` 的线序列目标：

```json
{"lines":[...]}
```

也就是说：

- 训练目标目前是 `road-line map`
- `Intersection.geojson` 目前不会进入 `v1.0` 训练目标

这是刻意做的第一版最小适配，因为 `v1.0` 的 patch-only baseline 本来就是线图风格，不是 lane+intersection 双任务风格。

## 3. 数据集根目录

默认数据集根目录已经写成：

```bash
/home/zsy/Downloads/dataset-extracted
```

如果你机器上路径不同，可以运行前覆盖：

```bash
export DATASET_ROOT=/your/path/to/dataset-extracted
```

## 4. 输出目录

所有导出和训练输出默认都保存在：

```bash
unimapgen-v1.0/outputs/geo_patch_v1_adapter
```

也就是：

- 导出的 `train.jsonl / val.jsonl`
- 导出的 `images/`
- 生成的 `dataset_info.json`
- LLaMAFactory 训练输出

都会落在 `unimapgen-v1.0` 目录下面。

## 5. 依赖安装

建议单独准备一个 Python 3.10 环境。

### 基础环境

```bash
conda create -n unimapgen-v1 python=3.10 -y
conda activate unimapgen-v1
python -m pip install --upgrade pip
```

### PyTorch

下面命令需要按你的 CUDA 版本选择。

例如 CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你已经有可用的 GPU torch，可以跳过这一步。

### 适配器和训练依赖

```bash
pip install transformers peft accelerate datasets pillow numpy pyproj rasterio pyyaml
pip install llamafactory
```

安装完后建议确认：

```bash
llamafactory-cli --help
python -c "import rasterio, pyproj, transformers, peft; print('ok')"
```

## 6. 默认模型路径

一键脚本默认会使用：

```bash
$ROOT/../ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct
```

如果你本地模型不在这个位置，可以先覆盖：

```bash
export MODEL_PATH=/your/path/to/Qwen2___5-VL-3B-Instruct
```

## 7. 一键启动

进入 `unimapgen-v1.0` 目录后运行：

```bash
bash scripts/run_geo_patch_v1_lf_train.sh
```

这个脚本会自动做三件事：

1. 从当前 `GeoJSON + tif` 数据导出 `ShareGPT + images` 数据集
2. 生成一份适合 `v1.0` 的 LLaMAFactory 训练配置
3. 启动 LoRA SFT 训练

## 8. 常用覆盖参数

如果你想改默认路径或命令，可以在运行前覆盖这些环境变量：

```bash
export DATASET_ROOT=/home/zsy/Downloads/dataset-extracted
export OUTPUT_ROOT=/path/to/unimapgen-v1.0/outputs/geo_patch_v1_adapter
export MODEL_PATH=/path/to/Qwen2___5-VL-3B-Instruct
export PYTHON_BIN=python
export LLAMAFACTORY_BIN=llamafactory-cli
```

然后再执行：

```bash
bash scripts/run_geo_patch_v1_lf_train.sh
```

## 9. 导出结果怎么看

数据导出后，先看：

- `outputs/geo_patch_v1_adapter/train.jsonl`
- `outputs/geo_patch_v1_adapter/val.jsonl`
- `outputs/geo_patch_v1_adapter/images/`
- `outputs/geo_patch_v1_adapter/export_summary.json`

重点确认：

- 图像是不是已经从 `tif` 正常转成 `png`
- `assistant` 的目标是不是 `{"lines":[...]}` 结构
- `line_count` 是否正常

## 10. 我建议你第一次先做这个检查

第一次不要直接长训，先只跑导出：

```bash
python scripts/export_llamafactory_from_geo_patch_dataset.py \
  --dataset-root /home/zsy/Downloads/dataset-extracted \
  --output-root ./outputs/geo_patch_v1_adapter \
  --splits train val \
  --use-system-prompt
```

先确认 `jsonl` 和 `images` 正常，再跑一键训练。

## 11. 当前这条适配路线和你现在主线的关系

这条路线的目的不是替换你现在的主线，而是让你可以快速试一下：

- `v1.0` 这种线图式 patch-only baseline

在你当前数据上的表现。

但要注意：

- 它目前只适配 `Lane.geojson`
- 不包含你现在主线里的：
  - `Intersection` 双任务
  - UV GeoJSON 文本输出
  - companion reference
  - patch-state GeoJSON 续写

所以这条路线更像：

- 一个可比较的旧式 baseline

不是你当前主算法的完整替身。
