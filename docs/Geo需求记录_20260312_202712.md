# Geo Requirements Log 2026-03-12 20:27:12

## Polygon Coordinate Contract Clarification

- `Intersection.geojson` contains multiple `Feature` objects under one `FeatureCollection`.
- Each intersection polygon uses:
  - `"type": "Polygon"`
  - `"coordinates": [ [x, y, z], [x, y, z], ... ]`
- The expected output format is therefore the flattened point-list form after `"coordinates"`, not the standard GeoJSON ring-array form.
- Input files use English punctuation only. Earlier Chinese punctuation examples were illustrative, not the real dataset contract.

## Runtime Change Made

- Polygon reader remains tolerant:
  - it accepts both flat coordinate lists and ring-array style input
- Polygon writer now exports the flattened point-list form expected by the dataset/output contract
- Smoke config artifact export has been turned back on so train/eval/predict intermediate assets are emitted by default there as well

## Cut-Point Status

- Lane / linestring:
  - yes, clipped segments are turned into local target items and get `cut_in` / `cut_out` based on patch boundary side
- Intersection / polygon:
  - polygons are clipped before training/inference state construction
  - but there is not yet an explicit polygon `cut_in` / `cut_out` label the way lines have
  - current polygon continuity is carried mainly by boundary state anchors rather than polygon cut labels
