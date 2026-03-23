# Geo Requirements Log 20260312_172817

## Purpose
This document records the current user requirements, data assumptions, model constraints, and runtime/export expectations for the `UniMapGenStrongBaseline` geo pipeline as of `2026-03-12 17:28:17`.

## Final Deliverables
- Final prediction outputs must be `.geojson` files.
- Training supervision files are also `.geojson` files.
- The pipeline must support:
  - training
  - validation/evaluation
  - inference/prediction
- Intermediate artifacts must also be saved during training and inference, including:
  - kept patches
  - discarded patches
  - predicted map `.geojson` snapshots that would otherwise only live in memory

## Dataset Structure
Current expected dataset layout:

```text
data/geo_vector_dataset/
  train/
    <sample_id>/
      patch_tif/
        0.tif
        0_edit_poly.tif
      label_check_crop/
        Lane.geojson
        Intersection.geojson
  val/
    <sample_id>/
      patch_tif/
        0.tif
        0_edit_poly.tif
      label_check_crop/
        Lane.geojson
        Intersection.geojson
  test/
    <sample_id>/
      patch_tif/
        0.tif
        0_edit_poly.tif
      label_check_crop/
        Lane.geojson
        Intersection.geojson
```

Notes:
- `test` can omit labels for pure prediction.
- `eval` requires labels.
- File paths are configured in `configs/geo_vector_lora.yaml` and `configs/geo_vector_full.yaml`.

## GeoTIFF Information Provided By User
The user provided these GeoTIFF facts and they are treated as requirements/assumptions:
- Input image format: GeoTIFF (`.tif`)
- CRS example: `EPSG:32650` / `WGS 84 / UTM zone 50N`
- `Origin` is different for each tif
- `Pixel Size` is consistent:
  - `0.2000000000000000111`
  - `-0.2000000000000000111`
- Band order used by the current pipeline: `1, 2, 3`
- Per-band data range is `0..255`
- `Scale=1`, `Offset=0`
- `statistics_valid_percent=100`
- Example per-band statistics were provided by the user and are treated as observational context, not hard-coded normalization targets

Implications:
- The code must not assume a fixed origin.
- CRS / affine transform must be read per image.
- Coordinate conversion must be image-specific.

## Output Contract Requirements
- `Lane.geojson` must be a `FeatureCollection` of `LineString` lane objects.
- `Intersection.geojson` must be a `FeatureCollection` of `Polygon` intersection objects.
- Intersection polygon output must be closed in final GeoJSON:
  - the first and last coordinate must be identical in the exported ring
- The examples shown by the user are incomplete examples only.
- The code must not hard-code:
  - a fixed lane count
  - a fixed intersection count
  - a fixed property schema beyond what the training data teaches

## Modeling Requirements
- The user explicitly requires this coordinate chain:

```text
GeoJSON -> pixel uv -> quantized coordinate tokens -> model -> dequantize -> GeoJSON
```

- The user does not want the model trained as a free-form GeoJSON text generator.
- The user wants structured coordinate generation, with GeoJSON as post-processing/output formatting.
- `state update` must focus on:
  - cut-point anchors
  - local incremental generation
  - not full previous-text continuation

## Current Implemented Architecture
- Visual encoder: DINO-based satellite encoder
- Language/decoder backbone: Qwen causal LM
- Input to model:
  - image patch
  - task prompt
  - cut-point state anchors from left/top boundary
- Output from model:
  - structured quantized coordinate tokens
  - properties currently remain compact JSON spans inside the structured sequence

Current coordinate flow:

```text
GeoJSON
-> project to raster CRS / absolute pixel coordinates
-> crop to patch
-> optional 6m resampling
-> patch-local uv
-> quantized coordinate tokens
-> model
-> dequantize uv
-> absolute pixel coordinates
-> GeoJSON export
```

## State Update Requirements
- Cross-patch prediction must use boundary-local state, not full prior-text replay.
- Current state input is built from cut-point anchor segments on:
  - left border
  - top border
- State is local and incremental.

## Tiling / Patch Requirements
- Original tif sizes vary greatly.
- Some images may be very wide and short.
- Some images may be very tall and narrow.
- Tiling/cropping must therefore support strongly non-square source images.
- Current tiling is sliding-window based and does not require square full images.

Current default tiling-related settings:
- `tile_size_px = 1024`
- `overlap_px = 256`
- `keep_margin_px = 128`
- `state_update.border_margin_px = 128`
- `image_size = 448`

`image_size` is the model input canvas size after resize/pad, not the original tif size.

## Resampling Requirement
- Fixed resampling interval requested by user: `6 meters`
- Sharp turns / corners do not need to perfectly preserve 6m spacing

Current implementation:
- Converts `6m` into pixels using the current raster pixel size
- Performs geometric resampling in pixel space

## Context-Length Concern
- The user explicitly raised concern about large maps exceeding model context.
- Requirement:
  - avoid feeding whole large maps as one sequence
  - rely on tiling + state update + local incremental generation

Current mitigations:
- tile-based training and inference
- state anchors limited to local border regions
- token caps:
  - prompt tokens
  - state tokens
  - target tokens
  - max generated tokens

## Reordering / Canonicalization Requirement
- The user asked whether `.geojson` reordering is performed.
- Current behavior:
  - line direction is canonicalized
  - polygon start vertex is canonicalized
  - property JSON is key-sorted when compacted
- Not yet implemented:
  - full global feature ordering across the entire map graph

## Intermediate Artifact Export Requirement
The user requested persistent intermediate outputs instead of dropping them after loss computation or after merge.

Current implementation now exports:
- patch audit metadata
- kept patch images
- discarded patch images
- resized patch previews
- train batch snapshots
  - prompt
  - state items
  - target items
  - GT GeoJSON
  - predicted GeoJSON
- val batch snapshots
  - same as train snapshots
- predict-time tile-level GeoJSON
  - raw per-tile prediction GeoJSON
  - kept per-tile GeoJSON
- eval-time sample artifacts
  - predicted GeoJSON
  - GT GeoJSON
  - patch audit

## Training Mode Requirement
- `LoRA` vs `full` must remain user-configurable in config/scripts.
- The code must not hard-wire training mode.

## Prompt Requirements
- Prompts must explicitly say:
  - do not assume a fixed count from examples
  - do not assume a fixed property template from examples
  - intersection polygons must end in a closed ring in final GeoJSON

## Important Non-Requirements / Things Not Hard-Coded
- No fixed lane count
- No fixed intersection count
- No single example property list treated as exhaustive
- No fixed tif origin
- No assumption that the whole map is square

## Open Technical Limits Still Present
- Properties are still carried as compact JSON spans, not fully discretized schema tokens.
- Full graph-level feature reordering is still not implemented.
- Artifact export can produce a large amount of files on big datasets and may need throttling through script parameters.
