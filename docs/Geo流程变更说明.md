# Geo Pipeline Changes

## Added Modules

- [unimapgen/geo/schema.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/geo/schema.py)
  Loads task schemas for `lane` and `intersection`, including geometry type, prompts and property fields.

- [unimapgen/geo/tokenizer.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/geo/tokenizer.py)
  Implements the custom structured tokenizer:
  - GeoJSON properties to tokens
  - coordinate quantization
  - grammar-constrained decoding support
  - Qwen tokenizer extension

- [unimapgen/geo/io.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/geo/io.py)
  Handles GeoTIFF reading, CRS conversion, GeoJSON parsing, and pixel/world/GeoJSON conversion.

- [unimapgen/geo/geometry.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/geo/geometry.py)
  Handles review-mask crop bbox, square letterbox resize, tile window generation, point transforms and geometry resampling.

- [unimapgen/geo/dataset.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/geo/dataset.py)
  Implements the new dataset and collator for:
  - `patch_tif/0.tif`
  - `patch_tif/0_edit_poly.tif`
  - `Lane.geojson`
  - `Intersection.geojson`
  Training now supports large-image tiling:
  - tiles are generated from the reviewed area
  - each tile is read from the GeoTIFF by window
  - labels are clipped to the tile before tokenization

- [unimapgen/geo/metrics.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/geo/metrics.py)
  Implements engineering-style evaluation metrics:
  - lane precision / recall / F1 at 2m
  - lane mean Hausdorff distance
  - lane endpoint error
  - lane property exact accuracy
  - intersection precision / recall / F1 at IoU 0.3
  - intersection mean IoU
  - intersection property exact accuracy
  - parse success rate
  - tile result deduplication after sliding-window inference

- [unimapgen/geo/pipeline.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/geo/pipeline.py)
  Shared builder and checkpoint logic for the new geo pipeline.

## Added Model

- [unimapgen/models/qwen_geo_generator.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/models/qwen_geo_generator.py)
  New model wrapper for:
  - frozen DINOv2 encoder
  - trainable modality projector
  - Qwen full fine-tune or LoRA mode
  - structured-token generation

## Added Scripts

- [unimapgen/train_geo_model.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/train_geo_model.py)
  Shared trainer with checkpoint resume and timestamped output directory support.

- [unimapgen/train_geo_lora.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/train_geo_lora.py)
  Entry point for frozen DINOv2 + projector + Qwen LoRA.

- [unimapgen/train_geo_full.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/train_geo_full.py)
  Entry point for frozen DINOv2 + projector + Qwen full fine-tune.

- [unimapgen/predict_geo_vector.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/predict_geo_vector.py)
  Supports:
  - single tif inference
  - folder batch inference
  - configured split inference
  Current inference uses sliding-window tiling on the whole tif, then merges and deduplicates outputs.
  Outputs `Lane.geojson` and `Intersection.geojson` per sample.

- [unimapgen/eval_geo_vector.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/eval_geo_vector.py)
  Runs end-to-end whole-image tiled prediction and metric aggregation on a split.

- [unimapgen/check_geo_data.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/check_geo_data.py)
  Checks sample paths, raster metadata and label file existence.

- [unimapgen/tokenize_geo_vector.py](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/unimapgen/tokenize_geo_vector.py)
  Converts one GeoJSON file into structured custom tokens for tokenizer inspection.

## Added Configs

- [configs/geo_vector_lora.yaml](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/configs/geo_vector_lora.yaml)
  Example config for LoRA training.

- [configs/geo_vector_full.yaml](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/configs/geo_vector_full.yaml)
  Example config for full Qwen fine-tuning.

## Usage

Tile sizes and overlaps are configured in:

- [configs/geo_vector_lora.yaml](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/configs/geo_vector_lora.yaml)
- [configs/geo_vector_full.yaml](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/configs/geo_vector_full.yaml)

Main knobs:

- `tiling.train.tile_size_px`
- `tiling.train.overlap_px`
- `tiling.eval.tile_size_px`
- `tiling.predict.tile_size_px`
- `postprocess.line_dedup_distance_m`
- `postprocess.polygon_dedup_iou`

Check data:

```bash
python -m unimapgen.check_geo_data --config configs/geo_vector_lora.yaml
```

Inspect tokenizer output for one file:

```bash
python -m unimapgen.tokenize_geo_vector --config configs/geo_vector_lora.yaml --task lane --image path/to/0.tif --geojson path/to/Lane.geojson
```

Train with LoRA:

```bash
python -m unimapgen.train_geo_lora --config configs/geo_vector_lora.yaml
```

Train with full Qwen fine-tune:

```bash
python -m unimapgen.train_geo_full --config configs/geo_vector_full.yaml
```

Batch inference on a directory:

```bash
python -m unimapgen.predict_geo_vector --config configs/geo_vector_lora.yaml --checkpoint outputs/geo_vector_lora/20260310_120000/best.pt --input_dir /data/tif_dir
```

Evaluate a split:

```bash
python -m unimapgen.eval_geo_vector --config configs/geo_vector_lora.yaml --checkpoint outputs/geo_vector_lora/20260310_120000/best.pt --split val
```

Ubuntu/Linux bash wrappers are available:

- [scripts/run_geo_lora_train.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_lora_train.sh)
- [scripts/run_geo_full_train.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_full_train.sh)
- [scripts/run_geo_eval.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_eval.sh)
- [scripts/run_geo_predict.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_predict.sh)

Windows PowerShell wrappers are also available:

- [scripts/run_geo_lora_train.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_lora_train.ps1)
- [scripts/run_geo_full_train.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_full_train.ps1)
- [scripts/run_geo_eval.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_eval.ps1)
- [scripts/run_geo_predict.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_predict.ps1)
