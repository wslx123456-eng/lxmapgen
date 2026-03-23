# 从 Patch-Only 母集到 16fix 训练版数据集指南


## 文档目标

文档面向这样一种情况：

- 你**已经有了导出的 patch-only 母集**
- 现在只想在这个基础上，继续构建 `16fix` 训练版数据集



## 一句话总结

如果你已经有了 `patch-only` 母集，那么从它到 `16fix` 训练版的主链实际上只需要 **一个核心脚本**：

- [build_patch_only_fixed_grid_targetbox_dataset.py]

你只是在做这件事：

- 把每个已有的 `896x896 patch` 展开成 `4x4=16` 个固定 target box 样本
- 图像仍然是整张 patch
- assistant target 改成只输出当前 box 内的路网
- 并把 empty box 下采样到约 `10%`

## 前提条件

你已经有一个标准 patch-only 母集，目录结构大致应为：

```text
/your/data/outputs/paper16_patch_only_100img_system/
  train.jsonl
  val.jsonl
  meta_train.jsonl
  meta_val.jsonl
  images/train/...
  images/val/...
  dataset_info.json
```

如果你的目录里至少具备这些文件，就可以继续做 `16fix`：

- `train.jsonl`
- `meta_train.jsonl`
- `images/...`

如果你还想一起构建验证版，也最好同时有：

- `val.jsonl`
- `meta_val.jsonl`

## 核心脚本

主脚本：

- [build_patch_only_fixed_grid_targetbox_dataset.py]

仓库里的典型 sbatch 模板：

- 训练版 16fix：
  - [build_patch_only_100img_fixed16_empty10.sbatch]
- 全框评估版 16fix：
  - [build_patch_only_100img_fixed16_allboxes_system.sbatch]

## 你到底在做什么

`16fix` 不是重新裁图，而是：

- 原来 patch-only 的每个样本对应一张 `896x896` patch
- 现在把这张 patch 再按固定 `4x4` 网格划成 `16` 个 target box
- 每个 target box 生成一个新的训练样本

所以：

- 图像输入：还是整张 `896x896`
- prompt：新增 `target_xy` 和 `target_box`
- assistant：只输出这个 box 内的 `target_box_map`

## 训练版 `16fix empty10`

### 作用

训练版的目标是：

- 保留所有非空 box
- 只保留一部分空 box
- 让最终 empty ratio 大约为 `10%`

这是当前项目里最常见的 `16fix` 训练版形式。

### 典型命令

```bash
PROJECT_ROOT=/your/project/UniMapGen
PATCH_ONLY_ROOT=/your/data/outputs/paper16_patch_only_100img_system
OUT_ROOT=/your/data/outputs

python $PROJECT_ROOT/scripts/build_patch_only_fixed_grid_targetbox_dataset.py \
  --input-root $PATCH_ONLY_ROOT \
  --output-root $OUT_ROOT/paper16_patch_only_100img_fixed16_empty10_system \
  --splits train \
  --grid-size 4 \
  --target-empty-ratio 0.10 \
  --seed 42 \
  --resample-step-px 12.0 \
  --boundary-tol-px 2.5 \
  --use-system-prompt-from-source \
  --image-root-mode symlink
```

### 参数解释

- `--input-root`
  - patch-only 母集根目录
- `--output-root`
  - 16fix 训练版输出目录
- `--splits train`
  - 这里只构建训练集
- `--grid-size 4`
  - 表示每张 patch 按 `4x4` 切成 16 个 box
- `--target-empty-ratio 0.10`
  - 控制空框比例约为 10%
- `--seed 42`
  - 控制 empty 下采样的一致性
- `--resample-step-px 12.0`
  - 对线段重采样的步长
- `--boundary-tol-px 2.5`
  - 边界 cut 类型判定的容差
- `--use-system-prompt-from-source`
  - 复用 patch-only 母集原来的 system prompt
- `--image-root-mode symlink`
  - 输出数据集的 images 目录使用软链接方式指向原图，节省空间

### 主要输出

```text
paper16_patch_only_100img_fixed16_empty10_system/
  train.jsonl
  meta_train.jsonl
  images/...
  dataset_info.json
  build_summary.json
```

### 这一版的语义

- 每条样本仍然对应同一张 `896x896` 图
- 但每条样本只负责一个固定的 target box
- target 不再是完整 patch map，而是局部 `target_box_map`

## 如果你还想顺手构建评估版 `allboxes`

虽然你现在主要问的是训练版，但通常建议同时留一版评估版，因为后续 grouped eval 会用到。

### 作用

- 保留所有 target box
- 不下采样 empty box
- 适用于 grouped eval / merge-back

### 典型命令

```bash
PROJECT_ROOT=/your/project/UniMapGen
PATCH_ONLY_ROOT=/your/data/outputs/paper16_patch_only_100img_system
OUT_ROOT=/your/data/outputs

python $PROJECT_ROOT/scripts/build_patch_only_fixed_grid_targetbox_dataset.py \
  --input-root $PATCH_ONLY_ROOT \
  --output-root $OUT_ROOT/paper16_patch_only_100img_fixed16_allboxes_system \
  --splits train val \
  --grid-size 4 \
  --target-empty-ratio 1.0 \
  --seed 42 \
  --resample-step-px 12.0 \
  --boundary-tol-px 2.5 \
  --image-root-mode symlink
```

### 和训练版的区别

- `empty10`：
  - 训练用
  - 空框下采样
- `allboxes`：
  - 评估用
  - 空框全保留

## 推荐目录命名

如果你的 patch-only 母集已经存在，建议这样命名：

```bash
/your/data/outputs/paper16_patch_only_100img_system
/your/data/outputs/paper16_patch_only_100img_fixed16_empty10_system
/your/data/outputs/paper16_patch_only_100img_fixed16_allboxes_system
```

这样和项目当前命名保持一致，后续训练、评测脚本更容易复用。







## 输入输出关系

### 输入

- 你已有的 patch-only 母集

### 输出

- 训练版：
  - `paper16_patch_only_100img_fixed16_empty10_system`
- 可选评估版：
  - `paper16_patch_only_100img_fixed16_allboxes_system`

## 一定不要混淆的概念

### `patch-only`

- 一个样本负责整张 patch 的完整路网

### `16fix`

- 一个样本负责整张 patch 图像中的一个固定 box

### `empty10`

- 训练版
- 空框下采样到约 10%

### `allboxes`

- 评估版
- 所有 box 保留

## 最短可执行链路

如果你已经有 patch-only 母集，那么最短链路其实就是：

1. 跑 [build_patch_only_fixed_grid_targetbox_dataset.py]

2. 设置：
  - `--grid-size 4`
  - `--target-empty-ratio 0.10`
3. 输出到：
  - `paper16_patch_only_100img_fixed16_empty10_system`

也就是说：

- **从 patch-only 到 16fix 训练版，本质上就是一条脚本链**
- 不再需要回到原始 `4096` 图像那一层

## 推荐执行顺序

1. 先确认 patch-only 母集完整
2. 先跑 `16fix empty10` 训练版
3. 再按需要补 `16fix allboxes` 评估版
4. 再接训练和 grouped eval

## 适用范围

如果你的输入已经是任意 `paper16_patch_only_xxx_system` 母集，这套方法都可以直接套：

- 不一定必须是 `100img`
- 只要输入格式是标准 patch-only 母集
- 就可以直接从这一步开始构建 `16fix`
