# Geo Requirements Log 20260312_174337

This document updates the geo requirements record with the latest user-provided constraints.

## New User-Provided Requirements
- Every future requirement the user gives must be reflected in the related docs.
- Each sample provides two tif files semantically:
  - one full-map GeoTIFF image
  - one black/white GeoTIFF mask
- In the black/white mask:
  - white = human-reviewed trusted region
  - black = unreviewed / untrusted region
- Black can be used as a training mask.
- If a patch contains too much black area, it should be discarded instead of teaching the model from untrusted content.
- During training, road targets should only be learned from the trusted region.

## Dataset Semantics
The current dataset layout remains:

```text
<split>/<sample_id>/
  patch_tif/0.tif
  patch_tif/0_edit_poly.tif
  label_check_crop/Lane.geojson
  label_check_crop/Intersection.geojson
```

Semantic meaning:
- `patch_tif/0.tif`: full map tif for the sample
- `patch_tif/0_edit_poly.tif`: black/white trusted-region mask tif
- `label_check_crop/Lane.geojson`: lane supervision GeoJSON
- `label_check_crop/Intersection.geojson`: intersection supervision GeoJSON

## Current Trusted-Mask Behavior
The current implementation now does all of the following for training:
- crops around the review mask bounding box when enabled
- scores tile candidates by white-mask coverage
- does not fall back to all tiles if no tile meets the train mask threshold
- discards empty train tiles by default
- filters train features by the trusted mask

Training defaults now are:
- `tiling.train.allow_empty_tiles: false`
- `tiling.train.fallback_to_all_if_empty: false`
- `data.train_filter_features_by_review_mask: true`
- `data.feature_mask_min_inside_ratio: 0.5`

Important detail:
- for line features, the training pipeline now keeps only contiguous trusted point runs after resampling
- for polygon features, the current behavior is keep/drop by inside ratio, not exact mask-geometry clipping

This means the trusted-region requirement is implemented more strongly for roads/lines than for polygons.

## Empty Patch Discard
Answer to the user question: yes, training now discards empty tiles by default.

Specifically:
- train config now uses `allow_empty_tiles: false`
- if a tile has no intersecting target feature, it is skipped from the train dataset

Eval and predict remain configurable and are not forced into the same policy by default.

## Large Black-Region Patch Discard
Answer to the user question: yes, training now treats the black/white tif as a real trusted-content mask.

Current train behavior:
- candidate tiles are annotated with mask coverage
- tiles below the configured white-region coverage threshold are rejected
- unlike before, train selection does not automatically fall back to all tiles if no candidate passes

## Token To GeoJSON Path
Current Qwen output conversion path is:

```text
Qwen output token ids
-> decode structured map items
-> dequantize uv coordinates
-> map patch-local uv back to absolute pixel coordinates
-> merge / deduplicate features across tiles
-> convert pixel features to GeoJSON
-> save .geojson
```

Code path:
- token decode: `GeoCoordTokenizer.decode_map_items(...)`
- uv to absolute pixel reconstruction: `uv_items_to_abs_feature_records(...)`
- final GeoJSON conversion: `pixel_features_to_geojson(...)`

## What Is Still Not Hard-Coded
- feature count is not fixed from examples
- property keys are not fixed to one hand-written example
- tif origin is not fixed
- input tif size is not fixed

## Remaining Limit
- polygon trusted-region handling is still approximate compared with exact mask-shape clipping
- if exact polygon-by-mask clipping is needed later, it should be implemented explicitly

## Feature Count Requirement Update

Latest user requirement:

- do not impose a fixed lane/intersection feature-count cap
- if a patch/sample contains more objects, the pipeline should not truncate them just because of `max_features`

Current implementation update:

- `serialization.tasks.lane.max_features <= 0` now means no explicit lane-count cap
- `serialization.tasks.intersection.max_features <= 0` now means no explicit intersection-count cap
- startup scripts now default both values to `0`

Important caveat:

- this removes the explicit feature-count truncation
- it does not remove sequence/context limits such as:
  - `text.target_max_tokens`
  - `decode.max_new_tokens`
  - the underlying LLM context window

So:

- there is no longer a hard object-count cap from `max_features`
- but very dense patches can still be limited by token budget rather than by a feature counter
