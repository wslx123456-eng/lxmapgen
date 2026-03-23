# Geo 训练推理评估代码阅读文档

这份文档是给“带着别人一起读代码”准备的。目标不是重复使用说明，而是把我新增的 `GeoTIFF + Lane/Intersection GeoJSON` 训练、推理、评估链路按源码调用关系拆开讲清楚，并把关键入口挂到源码上。

## 1. 先看什么

推荐按下面顺序阅读，最容易建立整体感：

1. 配置和任务定义
   - [configs/geo_vector_lora.yaml](../configs/geo_vector_lora.yaml)
   - [unimapgen/geo/schema.py](../unimapgen/geo/schema.py)
2. 训练入口
   - [unimapgen/train_geo_lora.py](../unimapgen/train_geo_lora.py)
   - [unimapgen/train_geo_model.py](../unimapgen/train_geo_model.py)
3. 组件装配
   - [unimapgen/geo/pipeline.py](../unimapgen/geo/pipeline.py)
4. 数据读取和预处理
   - [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py)
   - [unimapgen/geo/io.py](../unimapgen/geo/io.py)
   - [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py)
5. token 化
   - [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py)
6. 模型前向和生成
   - [unimapgen/models/qwen_geo_generator.py](../unimapgen/models/qwen_geo_generator.py)
7. 推理与回拼
   - [unimapgen/geo/inference.py](../unimapgen/geo/inference.py)
   - [unimapgen/predict_geo_vector.py](../unimapgen/predict_geo_vector.py)
8. 评估
   - [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py)
   - [unimapgen/eval_geo_vector.py](../unimapgen/eval_geo_vector.py)

## 2. 整体调用链

### 2.1 训练

`train_geo_lora.py` 只是一个很薄的入口：

- [train_geo_lora.py](../unimapgen/train_geo_lora.py) 第 1 层只做 `argparse`
- 它调用 [run_training](../unimapgen/train_geo_model.py) 入口

实际训练主链路是：

`train_geo_lora.py` -> `run_training()` -> `build_geo_components()` -> `build_geo_dataset()` -> `DataLoader` -> `QwenSatelliteGeoGenerator.forward()` -> `loss.backward()` -> `save checkpoint`

关键源码：

- [unimapgen/train_geo_model.py](../unimapgen/train_geo_model.py) `run_training` 在第 52 行
- [unimapgen/geo/pipeline.py](../unimapgen/geo/pipeline.py) `build_geo_components` 在第 119 行
- [unimapgen/geo/pipeline.py](../unimapgen/geo/pipeline.py) `build_geo_dataset` 在第 61 行
- [unimapgen/models/qwen_geo_generator.py](../unimapgen/models/qwen_geo_generator.py) `forward` 在第 209 行

### 2.2 推理

推理入口是 [unimapgen/predict_geo_vector.py](../unimapgen/predict_geo_vector.py)：

- `_resolve_inputs` 第 17 行负责解析单图、目录或 split 输入
- `main` 第 51 行负责加载模型、跑滑窗推理、保存 `Lane.geojson` 和 `Intersection.geojson`

推理主链路是：

`predict_geo_vector.py` -> `_resolve_inputs()` -> `run_tiled_sample_prediction()` -> `model.generate()` -> `decode tokens` -> `transform_points_to_original()` -> `deduplicate_feature_records()` -> `pixel_features_to_geojson()`

关键源码：

- [unimapgen/geo/inference.py](../unimapgen/geo/inference.py) `run_tiled_sample_prediction` 第 30 行
- [unimapgen/models/qwen_geo_generator.py](../unimapgen/models/qwen_geo_generator.py) `generate` 第 248 行
- [unimapgen/geo/io.py](../unimapgen/geo/io.py) `pixel_features_to_geojson` 第 320 行
- [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py) `deduplicate_feature_records` 第 215 行

### 2.3 评估

评估入口是 [unimapgen/eval_geo_vector.py](../unimapgen/eval_geo_vector.py)：

- `_collect_samples` 第 45 行负责收集验证集样本
- `main` 第 63 行负责逐样本推理，再与真值比较

评估主链路是：

`eval_geo_vector.py` -> `_collect_samples()` -> `run_tiled_sample_prediction()` -> `geojson_to_pixel_features()` -> `evaluate_lane_predictions()` / `evaluate_intersection_predictions()` -> `save_json()`

关键源码：

- [unimapgen/geo/io.py](../unimapgen/geo/io.py) `geojson_to_pixel_features` 第 258 行
- [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py) `evaluate_lane_predictions` 第 67 行
- [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py) `evaluate_intersection_predictions` 第 99 行

## 3. 配置层如何控制整条链路

主配置是：

- [configs/geo_vector_lora.yaml](../configs/geo_vector_lora.yaml)
- [configs/geo_vector_full.yaml](../configs/geo_vector_full.yaml)

建议重点看这些配置块：

- `data`
  - 数据根目录、split 名、相对路径、`image_size`
  - `band_indices: [1, 2, 3]` 已按你的数据约定写成 `R/G/B`
- `serialization`
  - token 序列长度、坐标离散 bin 数、重采样间距
- `model`
  - DINOv2、Qwen、本地权重路径、LoRA 配置
- `train`
  - 学习率、epoch、resume、输出目录
- `decode`
  - 推理时 `max_new_tokens`、grammar 约束、采样策略
- `tiling`
  - 大图裁块训练、评估、推理的窗口大小和重叠
- `postprocess`
  - 预测结果去重阈值
- `evaluation`
  - lane 匹配距离阈值和 intersection 的 IoU 阈值

这里的语法是标准 YAML：

- `a: 1` 表示键值对
- `[1, 2, 3]` 表示列表
- `true/false` 表示布尔
- `null` 表示空值
- `${ENV:-default}` 是环境变量回退写法

例如：

```yaml
data:
  dataset_root: ${UNIMAPGEN_GEO_ROOT:-data/geo_vector_dataset}
```

表示优先读环境变量 `UNIMAPGEN_GEO_ROOT`，没有时才退回 `data/geo_vector_dataset`。

## 4. 任务定义和 prompt 在哪里

任务定义在 [unimapgen/geo/schema.py](../unimapgen/geo/schema.py)：

- `FieldSchema` 第 11 行
- `TaskSchema` 第 27 行
- `_default_task_specs` 第 43 行
- `load_task_schemas` 第 93 行

这里定义了两件事：

1. 每个任务的几何类型
   - `lane -> linestring`
   - `intersection -> polygon`

2. 每个任务要预测的属性字段
   - `Lane` 的 `Id`、`LaneType`、`Width` 等
   - `Intersection` 的 `Id`、`IntersectionType`、`IsRegular`、`IntersectionSubType`

prompt 不是写死在训练循环里，而是跟 `TaskSchema.prompt_template` 绑定。训练和推理都会通过模板生成任务提示词。

调用位置：

- [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) 第 202 行附近构造 `prompt_text`
- [unimapgen/geo/inference.py](../unimapgen/geo/inference.py) `prompt_text_for_task` 第 21 行

## 5. 数据预处理是怎么做的

### 5.1 读样本目录

[unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) 第 62 行开始是 `GeoVectorDataset`。

初始化时做这些事：

1. 进入 `dataset_root/split`
2. 枚举每个样本目录
3. 读取
   - `patch_tif/0.tif`
   - `patch_tif/0_edit_poly.tif`
   - `label_check_crop/Lane.geojson`
   - `label_check_crop/Intersection.geojson`
4. 用 `geojson_to_pixel_features()` 先把真值从经纬度转成原图像素坐标
5. 按 tile 把一个样本拆成多个训练 item

核心位置：

- [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) 第 69 行开始扫描样本
- [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) 第 108 行调用 `geojson_to_pixel_features`
- [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) 第 301 行 `_append_task_items`

### 5.2 读 tif 和 review mask

核心 IO 在 [unimapgen/geo/io.py](../unimapgen/geo/io.py)：

- `read_rgb_geotiff` 第 66 行
- `read_binary_mask` 第 96 行
- `read_raster_meta` 第 109 行

这里用到的关键语法：

- `with rasterio_open(path) as ds:` 是 Python 上下文管理器
  - 作用是打开文件后自动关闭
- `Window(...)` 是 `rasterio` 的窗口读取对象
  - 不是整图全读，而是按裁块窗口读

训练和推理的大图裁块都靠这个窗口读取。

### 5.3 审核区域裁剪

review mask 的白区先被算成一个 bbox。

核心函数：

- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `compute_mask_bbox` 第 115 行
- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `expand_bbox` 第 126 行

逻辑是：

1. 找出 mask 中所有非零像素
2. 求最小外接矩形
3. 再按 `review_crop_pad_px` 扩一圈

这一步不直接把裁剪结果落盘，它只是得到一个 `crop_bbox`，供后面按窗口读取 tif 使用。

### 5.4 大图分块

大图切 tile 的核心函数在 [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py)：

- `generate_tile_windows` 第 144 行
- `annotate_tile_windows_with_mask` 第 193 行
- `select_tile_windows` 第 222 行

`GeoVectorDataset._build_tile_windows()` 在 [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) 第 264 行把这几步串起来。

流程是：

1. 根据整图宽高或 review bbox 生成滑窗
2. 对每个滑窗统计白区占比和白区像素数
3. 过滤掉 review 覆盖太少的 tile
4. 留下一组训练窗口

这里有两个 bbox 概念：

- `bbox`
  - 真正读图的窗口
- `keep_bbox`
  - 在推理时用来决定“这个 tile 里哪些目标应该保留”的中心区域

`keep_bbox` 的作用是减少重叠 tile 之间的重复目标。

### 5.5 GeoJSON 坐标转原图像素坐标

GeoJSON 转像素坐标的核心在 [unimapgen/geo/io.py](../unimapgen/geo/io.py)：

- `detect_geojson_crs` 第 145 行
- `_build_transformer` 第 154 行
- `_project_coords` 第 169 行
- `_world_to_pixel` 第 186 行
- `geojson_to_pixel_features` 第 258 行

这一步的实际流程是：

1. 从 GeoJSON 里读出 CRS
2. 用 `pyproj.Transformer` 从 `CRS84` 重投影到 tif 的 CRS
3. 用 tif 的 affine transform 把世界坐标转成像素坐标

注意这里不是直接把经纬度喂给模型。模型看到的是像素坐标。

### 5.6 道路坐标顺序、闭合点和重采样

这一块最容易被误读，所以单独说明。

#### 当前实现做了什么

1. 保留原始几何点序
   - `LineString` 的点顺序默认沿用 GeoJSON 原顺序
   - 没有做“重新排序成最短路径”或“按道路拓扑重建顺序”

2. Polygon 在读入时会去掉重复闭合点
   - [unimapgen/geo/io.py](../unimapgen/geo/io.py) `_extract_pixel_points` 第 300 行
   - 如果首尾点相同，会先去掉最后一个重复点

3. Polygon 在导出 GeoJSON 时会重新补回闭合点
   - [unimapgen/geo/io.py](../unimapgen/geo/io.py) `pixel_features_to_geojson` 第 320 行

4. 几何点会按固定间距重采样
   - [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `_resample_path` 第 299 行
   - [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `resample_feature_points` 第 338 行

#### 当前实现没有做什么

- 没有做 lane graph 拓扑重建
- 没有做道路方向自动纠正
- 没有做线段拼接/断裂修复
- 没有做“把乱序点重新排成一条正确道路线”的专门算法

所以如果别人问“道路坐标重排序代码在哪”，准确回答应该是：

- 当前代码只做“保留原顺序 + 重采样”
- 没有做复杂的道路重排序

#### 重采样间距怎么来的

在 [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) `__getitem__` 第 147 行里：

1. 先取原图像素分辨率
2. 再把米制间距换算成原图像素间距
3. 再根据 resize 缩放到模型输入坐标
4. 最后调用 `resample_feature_points()`

核心公式是：

```text
interval_px = sample_interval_meter / pixel_size_meter
interval_model_px = interval_px * scale_mean
```

其中：

- `sample_interval_meter` 来自配置 `serialization.sample_interval_meter`
- `pixel_size_meter` 来自 tif 的 geotransform
- `scale_mean` 是裁剪后再缩放到模型输入尺寸时的平均缩放系数

这意味着：

- 你在配置里改的是“现实世界的采样间距”
- 代码会自动映射到 tile 和模型尺寸上

### 5.7 裁块后的几何裁剪

训练时并不是简单丢掉 bbox 外的整条线，而是会把几何裁到窗口里。

核心函数：

- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `feature_intersects_bbox` 第 399 行
- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `clip_feature_to_bbox` 第 411 行
- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `_shapely_to_point_arrays` 第 480 行

这里用到 `shapely`：

- `LineString`
- `Polygon`
- `box`

也就是先把几何和 bbox 变成几何对象，再做裁剪交集。

### 5.8 resize 到模型输入坐标

裁完 tile 后还要缩放到 `image_size x image_size`。

核心函数：

- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `build_resize_context` 第 241 行
- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `transform_points_to_model` 第 272 行
- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `transform_points_to_original` 第 281 行

`ResizeContext` 是个 `@dataclass`，用于保存：

- 原图宽高
- crop 起点
- crop 尺寸
- resize 后尺寸
- pad 偏移

这部分是训练和推理都共用的几何变换上下文。

### 5.9 数据增强

训练增强在 [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py)：

- `_maybe_augment` 第 250 行

底层实现：

- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `apply_square_augment` 第 355 行

当前支持：

- 90 度旋转
- 水平翻转
- 垂直翻转

这里重要的一点是：图像和几何点会一起变换，保证监督仍然对齐。

## 6. token 化是怎么设计的

tokenizer 在 [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py)。

分两层：

1. `GeoVectorTokenizer`
   - 自定义结构化 token 词表
   - 负责把属性和坐标编码成保留 token
2. `QwenGeoTokenizer`
   - 把自定义 token 和 Qwen 原词表拼在一起
   - 负责 prompt token 和 map token 的双向转换

关键入口：

- [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py) `encode_task_features` 第 115 行
- [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py) `decode_task_features` 第 188 行
- [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py) `valid_next_token_ids` 第 309 行
- [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py) `encode_prompt` 第 602 行
- [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py) `encode_map_token_ids` 第 608 行
- [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py) `decode_qwen_map_ids_to_custom_ids` 第 611 行
- [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py) `valid_next_qwen_map_ids` 第 619 行

### 6.1 自定义 token 结构

你会看到词表里有这几类 token：

- 结构 token
  - `<bos> <eos> <obj> <props> <coords> <geom_end> <obj_end>`
- 类型 token
  - `<type_string> <type_int> <type_float> <type_bool> <type_null>`
- 真假值 token
  - `<bool_true> <bool_false>`
- 属性名 token
  - `<key_Id> <key_LaneType> ...`
- 坐标 token
  - `<x_i> <y_i>`
- 字节 token
  - `<b_00> ... <b_ff>`

为什么属性值要走字节 token：

- string / int / float 最终都可以稳定转成字节序列
- 不依赖 Qwen 原生分词结果
- 结构更可控

### 6.2 坐标 token 化

几何点不会直接作为浮点文本输出，而是被量化到 `coord_num_bins` 个离散 bin。

这一步的好处是：

- 词表封闭
- 解码稳定
- 训练目标固定成离散 token 分类问题

### 6.3 grammar 约束

`valid_next_token_ids()` 和 `valid_next_qwen_map_ids()` 负责语法约束。

推理时如果开启 `use_grammar_constraint: true`，模型每一步只允许在“当前语法状态合法”的 token 里选下一个 token。

这能显著减少：

- 漏掉 `<obj_end>`
- 属性字段顺序错乱
- 坐标 token 和属性 token 混写

## 7. collator 和 batch 组织

collator 在 [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py)：

- `GeoVectorCollator` 第 365 行
- `__call__` 第 371 行

它做三件事：

1. stack 图像张量
2. 把每条样本的 prompt 编码成 Qwen token
3. 把自定义 map token 再映射成 Qwen 扩展词表中的 token id

batch 最终包含：

- `image`
- `prompt_input_ids`
- `prompt_attention_mask`
- `map_input_ids`
- `map_attention_mask`
- 以及一批调试和回溯信息，例如 `sample_ids`、`crop_bboxes`、`tile_windows`

## 8. 模型组件和前向

模型装配在 [unimapgen/geo/pipeline.py](../unimapgen/geo/pipeline.py) `build_geo_components` 第 119 行。

真正的模型是 [unimapgen/models/qwen_geo_generator.py](../unimapgen/models/qwen_geo_generator.py) 的 `QwenSatelliteGeoGenerator`：

- `__init__` 第 35 行
- `semantic_initialize_new_embeddings` 第 158 行
- `encode_prefix` 第 184 行
- `forward` 第 209 行
- `generate` 第 248 行

### 8.1 结构

模型可以拆成三块：

1. `SatelliteEncoder`
   - 读 RGB 图像
   - 由 DINOv2 提供视觉 token
2. `sat_proj`
   - 把 DINO token 投影到 Qwen hidden size
3. `Qwen`
   - 接 prompt 和 map token 做自回归建模

### 8.2 冻结与训练模式

这里支持三种模式：

- `freeze`
  - 冻结整个 LLM
- `lora`
  - Qwen 走 LoRA
- `full`
  - Qwen 全量训练

当前你的主配置是：

- 冻结 DINOv2
- Qwen 走 LoRA

### 8.3 loss 是怎么计算的

`forward()` 里做的是标准 Causal LM 交叉熵。

核心逻辑：

1. prefix = `satellite tokens + prompt tokens`
2. target = `map tokens`
3. prefix 对应的 label 全部设成 `-100`
4. 只有 map token 参与交叉熵

所以当前 loss 不是几何回归 loss，而是“结构化 map token 的 next-token cross entropy”。

源码位置：

- [unimapgen/models/qwen_geo_generator.py](../unimapgen/models/qwen_geo_generator.py) `forward` 第 209 行

### 8.4 generate 做了什么

`generate()` 是手写的逐步解码循环，不是直接调用 HF 的 `generate()`。

这样做的原因是要插入：

- grammar 约束
- allowed token 白名单
- repetition penalty
- top-k 或 greedy 解码

## 9. 训练循环怎么跑

训练主循环在 [unimapgen/train_geo_model.py](../unimapgen/train_geo_model.py)：

- `run_val` 第 26 行
- `run_training` 第 52 行

### 9.1 主流程

`run_training()` 里依次做：

1. 读 YAML 配置
2. 选择训练模式
3. 创建输出目录
4. 保存配置快照和运行元数据
5. `build_geo_components()`
6. `build_geo_dataset()` 构建 train/val
7. 加载 checkpoint
8. 创建 `DataLoader`
9. 创建 `AdamW`
10. 恢复 optimizer/scaler 状态
11. 进入 epoch 循环
12. 每轮跑验证，保存 `latest.pt` 和 `best.pt`

### 9.2 学习率

学习率调度来自 [unimapgen/utils.py](../unimapgen/utils.py)：

- `cosine_lr` 第 110 行

训练循环每 step 更新一次 lr，调用位置：

- [unimapgen/train_geo_model.py](../unimapgen/train_geo_model.py) 第 149 行附近

调度方式是：

- 前 `warmup_steps` 线性 warmup
- 之后 cosine decay

### 9.3 checkpoint 组织

checkpoint 相关函数在 [unimapgen/geo/pipeline.py](../unimapgen/geo/pipeline.py)：

- `maybe_load_model_checkpoint` 第 173 行
- `maybe_resume_training_state` 第 194 行
- `make_output_dir` 第 222 行
- `atomic_torch_save` 第 254 行
- `build_checkpoint_obj` 第 267 行

这里支持：

- 第一次训练按时间戳建目录
- 指定 `init_checkpoint` 加载已有权重
- `resume_in_place: true` 时继续写回原目录

## 10. 推理时大图怎么处理

推理核心在 [unimapgen/geo/inference.py](../unimapgen/geo/inference.py)：

- `run_tiled_sample_prediction` 第 30 行

### 10.1 滑窗推理流程

1. 读原图元信息
2. 根据 `tiling.predict` 生成 tile windows
3. 对每个 tile：
   - 读取 tif 局部窗口
   - resize 到模型输入
   - 对 lane 跑一次 generate
   - 对 intersection 再跑一次 generate
4. 把 tile 内预测点从模型坐标还原到原图像素坐标
5. 用 `keep_bbox` 去掉重叠边缘重复结果
6. 全图级别再做一次去重合并

### 10.2 为什么要用 `keep_bbox`

如果相邻 tile 有 overlap，同一条 lane 会在两个 tile 都被看见。

代码用 `feature_center_inside_bbox()` 来只保留“中心点落在 keep 区域”的目标，避免大量重复。

源码位置：

- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `feature_center_inside_bbox` 第 436 行

### 10.3 去重

最终还会做一次几何级去重：

- lane 用 Hausdorff 距离
- intersection 用 IoU

源码位置：

- [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py) `deduplicate_feature_records` 第 215 行

## 11. GeoJSON 是怎么写回去的

回写逻辑在 [unimapgen/geo/io.py](../unimapgen/geo/io.py) `pixel_features_to_geojson` 第 320 行。

流程是：

1. 原图像素坐标 -> 世界坐标
2. tif CRS -> `CRS84`
3. 组装成 `FeatureCollection`
4. 按任务名写成：
   - `Lane.geojson`
   - `Intersection.geojson`

注意：

- Polygon 导出时会自动补闭合点
- 默认会带第三维 `z=0.0`

输出落盘入口在 [unimapgen/predict_geo_vector.py](../unimapgen/predict_geo_vector.py) 第 95 行附近。

## 12. 评估指标是怎么定义的

评估指标逻辑在 [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py)。

### 12.1 lane 指标

`evaluate_lane_predictions()` 第 67 行输出：

- `lane_precision_2m`
- `lane_recall_2m`
- `lane_f1_2m`
- `lane_mean_hausdorff_m`
- `lane_mean_endpoint_error_m`
- `lane_property_exact_acc`
- `lane_pred_count`
- `lane_gt_count`
- `lane_unmatched_pred`
- `lane_unmatched_gt`

这里的匹配是：

- 先两两算 Hausdorff 距离
- 再做 greedy matching
- 只有距离小于阈值的 pair 才算匹配

源码：

- `_greedy_match_lines` 第 125 行
- `_endpoint_error` 第 177 行

`_endpoint_error()` 特别值得注意：

- 它同时计算正向端点误差和反向端点误差
- 取两者更小的那个

也就是说，评估对线方向有一定鲁棒性。

### 12.2 intersection 指标

`evaluate_intersection_predictions()` 第 99 行输出：

- `intersection_precision_iou30`
- `intersection_recall_iou30`
- `intersection_f1_iou30`
- `intersection_mean_iou`
- `intersection_property_exact_acc`
- `intersection_pred_count`
- `intersection_gt_count`

这里的匹配依据是 polygon IoU。

### 12.3 review mask 过滤

评估时还支持用审核 mask 再过滤预测结果：

- [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py) `filter_features_by_review_mask` 第 12 行

这一步是为了保证只在人工审核白区内评估。

### 12.4 汇总

[unimapgen/eval_geo_vector.py](../unimapgen/eval_geo_vector.py)：

- `_mean_metrics` 第 24 行负责对数值指标求平均
- `main` 第 63 行把所有 sample 指标写进 `sample_metrics`

## 13. 调试和检查脚本

### 13.1 数据检查

[unimapgen/check_geo_data.py](../unimapgen/check_geo_data.py)

作用：

- 检查 split 目录是否存在
- 检查每个样本的图像、mask、label 是否存在
- 打印 raster meta 和 feature 数量

适合在正式训练前先跑一遍。

### 13.2 token 化检查

[unimapgen/tokenize_geo_vector.py](../unimapgen/tokenize_geo_vector.py)

作用：

- 给一张图和一份 GeoJSON
- 看 token 化结果到底长什么样
- 验证坐标变换和 tokenizer 是否符合预期

## 14. 错误码体系

错误边界在 [unimapgen/geo/errors.py](../unimapgen/geo/errors.py)：

- `GeoPipelineError` 第 7 行
- `raise_geo_error` 第 18 行
- `wrap_geo_error` 第 25 行
- `run_with_geo_error_boundary` 第 42 行

入口脚本基本都包了一层 `run_with_geo_error_boundary()`，所以终端里会直接输出：

```text
[GEO-1204] failed to load GeoJSON: ...
```

带别人读代码时，可以把错误码分成三类理解：

- `GEO-11xx`
  - 训练、输出目录、数据 split 这类流程问题
- `GEO-12xx`
  - tif、GeoJSON、CRS、几何转换
- `GEO-14xx`
  - 模型和 checkpoint
- `GEO-16xx`
  - 推理 tile 生成与解码

## 15. 阅读源码时会频繁遇到的 Python 语法

### 15.1 `@dataclass`

出现在：

- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `ResizeContext`
- [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py) `TileWindow`
- [unimapgen/geo/io.py](../unimapgen/geo/io.py) `RasterMeta`
- [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) `GeoVectorDatasetConfig`

作用是把“纯数据对象”写得更短，不需要手写大量 `__init__`。

### 15.2 类型标注

比如：

```python
def read_raster_meta(path: str) -> RasterMeta:
```

含义是：

- `path` 期望是字符串
- 返回值是 `RasterMeta`

这是为了让代码更容易读，也方便 IDE 补全和静态检查。

### 15.3 上下文管理器

比如：

```python
with rasterio_open(path) as ds:
```

作用是确保文件句柄安全关闭。

### 15.4 `torch.inference_mode()` 和 `torch.no_grad()`

出现位置：

- [unimapgen/train_geo_model.py](../unimapgen/train_geo_model.py) `run_val`
- [unimapgen/models/qwen_geo_generator.py](../unimapgen/models/qwen_geo_generator.py) `generate`

作用是关闭梯度，减少显存和计算开销。

### 15.5 `np.ndarray` 的 HWC / CHW

图像在不同阶段的形状不同：

- 读 tif 时通常是 `HWC`
- 喂 PyTorch 前转成 `CHW`

相关代码：

- [unimapgen/geo/io.py](../unimapgen/geo/io.py) `read_rgb_geotiff`
- [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) `_resize_cropped_image`
- [unimapgen/geo/inference.py](../unimapgen/geo/inference.py) `_resize_cropped_image`

### 15.6 `dict` 批处理风格

整个工程大量使用 `Dict` 作为 batch 和中间结果容器，而不是大量自定义类。

这样做的好处是：

- 灵活
- 好打印
- DataLoader 和 JSON 序列化更方便

## 16. 当前实现的边界

这部分带读时最好提前说清楚，避免别人以为代码已经做了比实际更多的事。

当前已经实现：

- GeoTIFF + CRS84 GeoJSON 训练
- review mask 裁剪
- 大图 tile 训练 / 推理 / 评估
- lane / intersection 双任务
- 全属性预测
- 结构化 token 化
- Qwen LoRA 和全参训练
- checkpoint 恢复
- GeoJSON 回写
- 工程化评估指标

当前还没有实现：

- lane graph 拓扑重建
- 复杂道路坐标重排序
- MultiLineString / MultiPolygon
- 带洞 polygon
- 几何辅助 loss
- 训练时自动导出裁剪中间产物到磁盘

## 17. 如果现场只讲 15 分钟，怎么讲

可以只抓这 7 个跳转点：

1. 配置
   - [configs/geo_vector_lora.yaml](../configs/geo_vector_lora.yaml)
2. 任务 schema
   - [unimapgen/geo/schema.py](../unimapgen/geo/schema.py)
3. 训练入口
   - [unimapgen/train_geo_model.py](../unimapgen/train_geo_model.py)
4. 数据预处理主函数
   - [unimapgen/geo/dataset.py](../unimapgen/geo/dataset.py) `__getitem__` 第 147 行
5. 几何裁剪和重采样
   - [unimapgen/geo/geometry.py](../unimapgen/geo/geometry.py)
6. token 化
   - [unimapgen/geo/tokenizer.py](../unimapgen/geo/tokenizer.py)
7. 模型前向和生成
   - [unimapgen/models/qwen_geo_generator.py](../unimapgen/models/qwen_geo_generator.py)

如果要再加两处，就补：

8. 推理回拼
   - [unimapgen/geo/inference.py](../unimapgen/geo/inference.py)
9. 评估指标
   - [unimapgen/geo/metrics.py](../unimapgen/geo/metrics.py)
