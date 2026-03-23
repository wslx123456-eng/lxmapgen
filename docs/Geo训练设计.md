# Geo Training Design

## 1. 当前训练链路在做什么

当前这套链路不是让模型自由生成一整段自然语言或随意拼 JSON，而是走一条受约束的结构化生成流程：

1. 读取 `GeoTIFF`，按配置选择 `band1=R, band2=G, band3=B`
2. 用 `rasterio` 读取地理参考信息，把 `Lane.geojson` / `Intersection.geojson` 从 `CRS84` 重投影到 tif 的投影坐标系
3. 再把矢量坐标映射到像素坐标
4. 训练时按大图 tile 切块，只在 `0_edit_poly.tif` 的白色审核区域内取监督
5. 把属性和几何编码成保留 token 序列
6. 用冻结的 DINOv2 编码图像，用 Qwen 做自回归解码
7. 推理时分别生成 `Lane.geojson` 和 `Intersection.geojson`

对应实现：
- `unimapgen/geo/io.py`
- `unimapgen/geo/geometry.py`
- `unimapgen/geo/dataset.py`
- `unimapgen/geo/tokenizer.py`
- `unimapgen/models/qwen_geo_generator.py`
- `unimapgen/geo/inference.py`

## 2. 提示词如何设计

提示词是任务级提示，不承担复杂语义压缩，主要作用是：

- 明确当前任务是 `lane` 还是 `intersection`
- 明确输出目标是“审核过的 geojson 内容”
- 明确必须使用保留的结构化 vector token
- 明确需要预测所有属性和对应几何
- 明确最大 feature 数和最大点数

当前默认提示词在配置里可改：
- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`

`lane` 默认提示词：

```text
You are given a GeoTIFF image embedding.
Generate the reviewed Lane.geojson content using only the reserved vector tokens.
Predict all lane properties and the lane centerline geometry.
Output at most {max_features} lane features and at most {max_points_per_feature} points per lane.
```

`intersection` 默认提示词：

```text
You are given a GeoTIFF image embedding.
Generate the reviewed Intersection.geojson content using only the reserved vector tokens.
Predict all intersection properties and the outer polygon geometry.
Output at most {max_features} intersection features and at most {max_points_per_feature} points per polygon.
```

设计原则：

- 提示词尽量短，不把 schema 全部塞进 prompt
- schema 的硬约束主要放在 tokenizer 和 grammar 里，不放在自然语言里赌模型理解
- lane 和 intersection 分两次生成，避免一个长序列里混合两种几何结构，降低解码不稳定性

## 3. Lane 和 Intersection 是如何生成的

当前不是一条序列同时产出 lane 和 intersection，而是同一张 tile 做两次生成：

1. 用 `lane` prompt 生成 lane token 序列
2. 用 `intersection` prompt 生成 intersection token 序列
3. 各自反解成结构化 feature
4. 从模型坐标恢复到原图像素坐标
5. 多 tile 结果做去重合并
6. 最后再转回两份 GeoJSON

这样做的原因很直接：

- `Lane` 是 `LineString`
- `Intersection` 是 `Polygon`
- 两者属性 schema 不同
- 分开生成更容易控长度、控语法、控后处理

当前“写死”的部分只有任务 schema，不是把文本答案写死：

- `lane -> LineString`
- `intersection -> Polygon`
- 各字段名、字段类型、是否可空
- 每类对象最大数量、每个对象最大点数

这些都在 `unimapgen/geo/schema.py` 和配置文件里，后面如果加 `road_sign`、`stop_line`、`crosswalk`，直接扩展 task schema 和 tokenizer 即可，不需要重写整个框架。

## 4. 为什么不是直接生成原始 GeoJSON 文本

因为直接生成原始 GeoJSON 文本在工程上不稳，主要问题有：

- 大量标点、括号、引号会增加无效生成负担
- 容易出现 JSON 不闭合、字段错位、坐标错位
- 数值和布尔值混在自由文本里，后处理复杂
- lane 和 polygon 的几何边界难以做强约束

所以当前实现采用结构化 token：

- 对象开始结束：`<obj> <obj_end>`
- 属性开始：`<props>`
- 几何开始结束：`<coords> <geom_end>`
- 值类型：`<type_string> <type_int> <type_float> <type_bool> <type_null>`
- 布尔值：`<bool_true> <bool_false>`
- 字段名：`<key_Id>` 这类 token
- 文本值：按 UTF-8 byte 编成 `<b_xx>`
- 坐标值：量化成 `<x_i>`、`<y_i>`

这部分在：
- `unimapgen/geo/tokenizer.py`

推理时还会用 grammar 限制下一 token 的可选集合，尽量避免解码出非法结构。

## 5. Loss 是如何设计的

当前 loss 很简单，但路径是闭环的：

- 图像先过 DINOv2 得到视觉 token
- 视觉 token 过 `sat_proj` 映射到 Qwen hidden size
- 再拼接 prompt token embedding
- 最后拼接目标 map token embedding
- Qwen 做标准 causal LM next-token prediction

训练目标就是单一的自回归交叉熵：

- prefix 部分不算 loss
- 只对目标结构化 token 序列计算 loss
- padding token 用 `-100` mask 掉

对应实现：
- `unimapgen/models/qwen_geo_generator.py`

更具体一点，当前 label 设计是：

- `satellite tokens + prompt tokens` 位置的 label 全部置成 `-100`
- `map_input_ids` 对应位置参与交叉熵
- 所以模型学的是“给定图像和任务提示，预测下一个结构化地图 token”

当前没有额外加这些 loss：

- 几何回归 L1/L2 loss
- 匹配式 set loss
- 属性分类辅助 loss
- 拓扑一致性 loss
- coverage / count loss

原因不是这些不重要，而是先把第一版训练链路做成稳定可跑、可恢复、可评估。后续如果你发现：

- 属性准确率低
- 长 polyline 容易断
- polygon 顶点顺序不稳

再按问题加辅助 loss 更合理。

## 6. DINOv2 是否支持这种 tif

结论分两层：

### 6.1 从“输入格式能不能喂进去”看

可以，只要满足：

- 3 通道
- 通道语义明确为 `RGB`
- 像素值 `0-255`

当前代码会先把 tif 读成普通的 `HWC RGB` 数组，再缩放到 `0-1`，之后按 DINOv2 的均值方差做归一化。对 DINOv2 来说，它看到的是一个标准三通道图像 tensor，不会直接处理 GeoTIFF 容器本身。

对应实现：
- `unimapgen/geo/io.py`
- `unimapgen/geo/dataset.py`
- `unimapgen/models/encoders/satellite_encoder.py`

### 6.2 从“预训练分布是不是就是这类遥感数据”看

不完全是。

DINOv2 本身不是针对这种带地理参考的 RGB GeoTIFF 专门预训练的，它不理解：

- CRS
- geotransform
- UTM 投影
- GeoJSON 坐标系统

这些信息都在 DINOv2 外部处理。DINOv2 只负责从 RGB 外观里提视觉特征。

所以更准确的说法是：

- 它支持这类 tif 经过预处理后的三通道视觉输入
- 但它不是“原生地理矢量生成模型”
- 是否足够好，要靠你这套下游训练来适配任务

## 7. 我写了哪些组件

### 7.1 数据与几何

- `unimapgen/geo/schema.py`
  任务 schema，定义 lane/intersection 的字段、几何类型、点数限制、提示词模板
- `unimapgen/geo/io.py`
  读取 tif、mask、geojson，做 CRS 处理和像素坐标转换
- `unimapgen/geo/geometry.py`
  大图 tile、review mask 裁剪、几何裁剪、点重采样、坐标变换、增强
- `unimapgen/geo/dataset.py`
  训练/验证数据集，按 tile 构造样本并序列化监督

### 7.2 序列化与 tokenizer

- `unimapgen/geo/tokenizer.py`
  自定义结构化 token、Qwen token 对接、grammar 约束、token 反解

### 7.3 模型

- `unimapgen/models/qwen_geo_generator.py`
  冻结 DINOv2、视觉到语言投影层、Qwen LoRA/全参训练、forward loss、generate

### 7.4 流水线与训练

- `unimapgen/geo/pipeline.py`
  组件装配、dataset/tokenizer/model 构建、checkpoint 恢复、输出目录管理
- `unimapgen/train_geo_model.py`
  共享训练主循环
- `unimapgen/train_geo_lora.py`
  冻结 DINOv2，只训练模态转化器和 Qwen LoRA
- `unimapgen/train_geo_full.py`
  冻结 DINOv2，只训练模态转化器和 Qwen 全参数

### 7.5 推理与评估

- `unimapgen/geo/inference.py`
  大图滑窗推理、双任务生成、tile 级结果回拼
- `unimapgen/geo/metrics.py`
  lane/intersection 去重和评估
- `unimapgen/predict_geo_vector.py`
  单图或多图推理，输出两份 GeoJSON
- `unimapgen/eval_geo_vector.py`
  多指标评估
- `unimapgen/check_geo_data.py`
  数据自检
- `unimapgen/tokenize_geo_vector.py`
  检查 geojson 到 token 的编码效果

### 7.6 配置与脚本

- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`
- `scripts/run_geo_lora_train.sh`
- `scripts/run_geo_full_train.sh`
- `scripts/run_geo_eval.sh`
- `scripts/run_geo_predict.sh`

## 8. 当前方案的边界

当前版本我有意保持简单，边界如下：

- 只支持 `lane=LineString`
- 只支持 `intersection=Polygon`
- 默认是一张图分两次解码，不是联合解码
- loss 只有 token-level 自回归交叉熵
- DINOv2 只吃 RGB 外观，不吃地理元数据
- tile 合并主要靠中心区保留和去重，不是图拓扑级融合

这些都不是永久限制，只是第一版先把训练、继续训练、推理、评估跑顺。

## 9. 如果下一步要继续增强，优先级建议

我建议按这个顺序扩：

1. 先跑通一轮训练，看 token loss、parse success rate、lane/intersection 指标
2. 如果长线段断裂严重，再加几何连续性或长度相关约束
3. 如果属性错得多，再加属性辅助 loss 或属性重权重
4. 如果 tile 边界处重复和断裂明显，再升级 merge 策略
5. 如果后续要加更多要素，再扩 task schema，而不是改成自由文本生成
