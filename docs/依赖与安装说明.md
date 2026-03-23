# Dependencies And Install

## Required Python Packages

This geo pipeline depends on:

- `torch`
- `transformers`
- `accelerate`
- `peft`
- `numpy`
- `Pillow`
- `PyYAML`
- `tqdm`
- `rasterio`
- `pyproj`
- `shapely`
- `safetensors`

The package list is also stored in [requirements-geo.txt](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/requirements-geo.txt).

## Target Platform

The main target platform is `Ubuntu/Linux`.

- Core training, inference and evaluation code is Python and is cross-platform.
- The earlier `.ps1` wrappers are only convenience wrappers for Windows terminals.
- Matching Ubuntu bash wrappers are now provided under `scripts/*.sh`.

## One-Line Install

If your Python environment already contains the correct CUDA build of PyTorch:

```bash
pip install -r requirements-geo.txt
```

If you still need to install PyTorch first, install the matching CUDA wheel from the official PyTorch index, then run:

```bash
pip install -r requirements-geo.txt
```

## Optional Local Checks

Verify the main GIS dependencies:

```bash
python -c "import rasterio, pyproj, shapely; print('geo deps ok')"
```

Verify the model stack:

```bash
python -c "import torch, transformers, peft; print('model deps ok')"
```

## Notes

- `rasterio` is used to read GeoTIFF metadata, CRS and affine transform.
- `pyproj` is used to convert `CRS84` GeoJSON coordinates into the GeoTIFF CRS and back.
- `shapely` is used by the evaluation script for line distance and polygon IoU metrics.
- `shapely` is also used for tile clipping and tiled-output deduplication.
- The code assumes the GeoJSON CRS metadata is reliable, which matches the current requirement from the user conversation.
- GeoTIFF band order is configured as `band 1 = red`, `band 2 = green`, `band 3 = blue`.
