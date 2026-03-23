# Chat Memory Log

## User Requirements Summary

- The project must support confidential `tif + geojson` training data without exposing raw files.
- Training input uses:
  - `patch_tif/0.tif`
  - `patch_tif/0_edit_poly.tif`
  - `label_check_crop/Lane.geojson`
  - `label_check_crop/Intersection.geojson`
- Inference uses only the image `0.tif`.
- `GeoJSON` metadata CRS is reliable and is `CRS84`.
- `GeoTIFF` CRS is `EPSG:32650 / WGS 84 UTM zone 50N`.
- GeoTIFF pixel size is fixed at `0.2, -0.2`.
- GeoTIFF `Origin` differs per sample.
- GeoTIFF band order is `band1=R`, `band2=G`, `band3=B`.
- Images are large and sample sizes vary, so the pipeline must support tiled training and tiled inference.
- The review mask and the image share the same width, height, CRS, origin and pixel size.
- The review mask is binary: white region is manually reviewed ground truth, black region is not supervised.
- The provided `Lane.geojson` already contains only reviewed truth.
- The provided `Intersection.geojson` also contains only reviewed truth.
- Current output target is two predicted texts per sample:
  - `Lane.geojson`
  - `Intersection.geojson`
- All listed properties must be predicted, not just geometry.
- `Lane` uses `LineString`.
- `Intersection` uses `Polygon`.
- `Intersection` polygon closes by repeating the first point as the last point.
- DINOv2 must stay frozen.
- Two training modes are required:
  - modality projector + Qwen LoRA
  - modality projector + Qwen full fine-tuning
- Inference must support:
  - single sample
  - multi sample
- Training and inference must load checkpoints from config.
- Resume training must restore:
  - model
  - optimizer
  - scaler
  - epoch
  - global step
- Output folders should be time-based by training start time, but the user may also manually copy or rename folders.
- Target runtime platform should be Ubuntu/Linux.

## Confirmed Property Schemas

Lane fields:

- `Id`: string
- `Length`: int
- `RoadId`: string
- `LaneType`: int
- `TurnType`: int
- `RoadIDSource`: int
- `TurnTypeSource`: int
- `LaneTypeSource`: int
- `IsThereStopLine`: bool
- `Width`: float
- `Source`: int
- `IsIntersectionInLane`: bool
- `IsIntersectionOutLane`: bool
- `IsThereRoadSplit`: bool
- `IsLeftmost`: bool
- `IsRightmost`: bool

Intersection fields:

- `Id`: string
- `IntersectionType`: int
- `IsRegular`: bool or null
- `IntersectionSubType`: int
