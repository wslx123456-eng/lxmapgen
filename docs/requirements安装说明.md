# requirements 安装说明

现在依赖分成两套：

- [requirements.data.txt](/c:/DevelopProject/VScode/geomapgen/requirements.data.txt)
- [requirements.train.txt](/c:/DevelopProject/VScode/geomapgen/requirements.train.txt)

## 1. 只做数据处理
如果你当前只是分机处理数据集、导出 Stage A / Stage B 数据，安装：

```bash
pip install -r requirements.data.txt
```

这套依赖覆盖：

- family manifest 构建
- 多机分片处理
- Stage A 数据导出
- Stage B 数据导出
- 合并处理结果

## 2. 做训练和 rollout
如果你要训练或 rollout，安装：

```bash
pip install -r requirements.train.txt
```

这套依赖在数据处理基础上额外包含：

- `torch==2.1.2+cu121`
- `torchvision==0.16.2+cu121`
- `torchaudio==2.1.2+cu121`
- `transformers`
- `peft`
- `accelerate`
- `datasets`
- `llamafactory`

## 3. 兼容范围
当前默认按：

- `PyTorch 2.1`
- `CUDA 12.1` wheel

来组织训练依赖。

如果你机器上已经有可用的 `torch 2.1.x` GPU 环境，也可以先装业务依赖，再按你的环境单独装 torch。

## 4. 推荐顺序
### 多机处理
```bash
pip install -r requirements.data.txt
```

### 合并后训练
```bash
pip install -r requirements.train.txt
```

## 5. 自检
训练前建议检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果要确认 `Qwen2.5-VL` 相关依赖也能导入，再跑：

```bash
python scripts/check_geo_current_v1_env.py --require-qwen25-vl
```
