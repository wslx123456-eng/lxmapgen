# CUDA 11.4 / PyTorch 1.13 兼容说明

## 这次做了什么
为了让当前 `v1.0` 适配链尽量兼容旧环境，我做了两类改动：

1. 训练和 rollout 不再写死 `bf16`
2. 新增 legacy 环境安装与自检脚本

## 已改的兼容点
### 训练脚本
- [run_geo_current_v1_stagea_train_1gpu.sh](/c:/DevelopProject/VScode/geomapgen/scripts/run_geo_current_v1_stagea_train_1gpu.sh)
- [run_geo_current_v1_stageb_train_1gpu.sh](/c:/DevelopProject/VScode/geomapgen/scripts/run_geo_current_v1_stageb_train_1gpu.sh)

现在会自动检测：

- `torch` 主版本
- `torch.cuda.is_available()`
- `torch.cuda.is_bf16_supported()`

如果环境像 `PyTorch 1.13` 这种老版本，脚本会自动切到：

- `bf16: false`
- `fp16: true`

8 卡脚本复用这两个单卡脚本，所以同样会走这套逻辑。

### rollout 脚本
- [rollout_predict_qwen2_5vl_from_geo_current_family_manifest.py](/c:/DevelopProject/VScode/geomapgen/scripts/rollout_predict_qwen2_5vl_from_geo_current_family_manifest.py)

新增了：

- `--precision auto|fp16|bf16|fp32`

默认 `auto` 行为：

- torch 2.x 且 GPU 支持 bf16：用 `bfloat16`
- torch 1.x 或不支持 bf16：退到 `float16`
- 没有 CUDA：退到 `float32`

## 新增的一键安装和自检
### 安装脚本
- [install_geo_current_v1_legacy_torch113.sh](/c:/DevelopProject/VScode/geomapgen/scripts/install_geo_current_v1_legacy_torch113.sh)

它会：

1. 创建 conda 环境
2. 安装 `PyTorch 1.13` 旧栈
3. 安装通用依赖
4. 可选安装 `LLaMAFactory`
5. 最后自动跑一遍 preflight

### 自检脚本
- [check_geo_current_v1_env.py](/c:/DevelopProject/VScode/geomapgen/scripts/check_geo_current_v1_env.py)

它会检查：

- Python 版本
- torch 版本
- CUDA 是否可用
- GPU 数量
- bf16 是否支持
- `transformers` 是否真的能导入 `Qwen2_5_VLForConditionalGeneration`
- `llamafactory-cli` 是否在 PATH 中

## 重要边界
这里要说清楚：我做的是**代码层兼容和运行时降级**，不是“保证任何 torch1.13 环境都能跑 Qwen2.5-VL”。

真正的硬边界在于：

- `Qwen2.5-VL` 需要较新的 `transformers` 运行支持
- 较新的 `transformers` 是否还能和 `torch1.13` 组合稳定工作，要看你实际装到的版本

所以现在最稳的判断方式不是猜，而是跑：

```bash
python scripts/check_geo_current_v1_env.py --require-qwen25-vl
```

如果这一步过了，再去跑训练。

## 推荐顺序
```bash
bash scripts/install_geo_current_v1_legacy_torch113.sh
conda activate geo-current-v1-torch113
python scripts/check_geo_current_v1_env.py --require-qwen25-vl
```

如果通过，再跑：

```bash
bash scripts/run_geo_current_v1_stagea_train_1gpu.sh
```

## 结论
现在代码已经尽量兼容 `CUDA 11.4 / PyTorch 1.13` 这类旧环境，主要体现在：

- 不再强制 `bf16`
- rollout 自动降精度
- 提供 legacy 一键安装
- 提供 preflight 自检

但 `Qwen2.5-VL` 本身是否能在你那套旧环境里真正运行，仍然要以后面的 import/preflight 结果为准。
