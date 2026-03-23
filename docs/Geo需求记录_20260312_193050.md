# Geo Requirements Log 2026-03-12 19:30:50

## This Update

- Property parsing must be tolerant. Invalid or loosely formatted property spans should not silently collapse to `{}` if they can be recovered; if they still cannot be parsed, raw text should be preserved instead of being dropped.
- Training and validation must use the review mask because only white-mask regions are trusted supervision.
- Prediction must run on the full image and must not use the review mask to restrict tiles.
- Polygon input/output must preserve all rings, not only the exterior ring.
- Lane/intersection feature count must not be hard-capped by `max_features`; `0` means no explicit feature-count cap.
- Token-length settings should use `0` to mean "do not proactively truncate at this layer".

## Effective Runtime Behavior After This Change

- `data.val_crop_to_review_mask = true`
- `data.eval_filter_features_by_review_mask = true`
- `tiling.eval.search_within_review_bbox = true`
- `predict` stage skips review-mask loading, so inference scans full-image tiles.
- `serialization.tasks.lane.max_features = 0`
- `serialization.tasks.intersection.max_features = 0`
- `text.prompt_max_tokens = 0`
- `text.state_max_tokens = 0`
- `text.target_max_tokens = 0`
- `decode.max_new_tokens = 0`
- `decode.min_new_tokens = 0`
- `decode.max_prop_tokens = 0`

## Data-Path Notes

- GeoJSON training labels are still the supervision source files.
- Internal geometry path is:
  `GeoJSON -> absolute pixel coords -> patch-local uv -> quantized coordinate tokens -> model -> dequantized uv -> absolute pixel coords -> GeoJSON`
- Polygon records now preserve all rings during:
  - GeoJSON read
  - patch clipping
  - uv conversion
  - token encode/decode
  - evaluation geometry conversion
  - GeoJSON export

## Remaining Practical Limit

- `max_features=0` removes explicit object-count truncation, but sequence length is still bounded by the model context window.
- `max_new_tokens=0` now means "use the remaining context budget", not mathematically infinite generation.
