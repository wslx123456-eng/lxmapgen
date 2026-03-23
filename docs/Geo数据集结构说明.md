# Geo Dataset Structure

## Recommended Layout

The new pipeline expects a split-based directory tree:

```text
dataset_root/
  train/
    sample_000001/
      patch_tif/
        0.tif
        0_edit_poly.tif
      label_check_crop/
        Lane.geojson
        Intersection.geojson
    sample_000002/
      ...
  val/
    sample_000101/
      patch_tif/
        0.tif
        0_edit_poly.tif
      label_check_crop/
        Lane.geojson
        Intersection.geojson
  test/
    sample_000201/
      patch_tif/
        0.tif
        0_edit_poly.tif
      label_check_crop/
        Lane.geojson
        Intersection.geojson
```

## Matching Config Fields

The example configs are already aligned to this structure:

- `data.image_relpath: patch_tif/0.tif`
- `data.review_mask_relpath: patch_tif/0_edit_poly.tif`
- `data.label_relpaths.lane: label_check_crop/Lane.geojson`
- `data.label_relpaths.intersection: label_check_crop/Intersection.geojson`

If your internal directory names differ, only change these config fields. The Python code does not hardcode your confidential numbering scheme.

## Important Conventions

- `0.tif` is the image used for training and inference.
- `0_edit_poly.tif` is the reviewed-area mask used during training and evaluation.
- `Lane.geojson` contains only reviewed lane truth.
- `Intersection.geojson` contains only reviewed intersection truth.
- `Lane.geojson` uses `LineString`.
- `Intersection.geojson` uses `Polygon`.
- GeoJSON CRS is expected to be reliable and currently uses `CRS84`.
- GeoTIFF CRS can differ from GeoJSON CRS. The pipeline converts GeoJSON coordinates into the GeoTIFF CRS automatically.
- Large images are handled by tile-based training and tile-based inference. Tile size and overlap are controlled from config.

## Split Strategy

You must create `train`, `val`, and optionally `test` folders yourself. The project does not infer splits from filenames.

## Large Confidential Datasets

If you do not want to physically move files, create lightweight split folders and place each sample under a split using your own copy, mirror, junction, or export workflow. The code only needs the final relative paths above to exist.
