import argparse
import json
from pathlib import Path

from geo_current_dataset_v1_common import (
    DEFAULT_IMAGE_RELPATH,
    DEFAULT_INTERSECTION_RELPATH,
    DEFAULT_LANE_RELPATH,
    DEFAULT_MASK_RELPATH,
    build_manifest_for_dataset,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1-style family manifest from the current GeoTIFF+GeoJSON dataset.")
    parser.add_argument("--dataset-root", type=str, default="/dataset/zsy/dataset-extracted")
    parser.add_argument("--train-root", type=str, default="")
    parser.add_argument("--val-root", type=str, default="")
    parser.add_argument("--output-manifest", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--image-relpath", type=str, default=DEFAULT_IMAGE_RELPATH)
    parser.add_argument("--mask-relpath", type=str, default=DEFAULT_MASK_RELPATH)
    parser.add_argument("--lane-relpath", type=str, default=DEFAULT_LANE_RELPATH)
    parser.add_argument("--intersection-relpath", type=str, default=DEFAULT_INTERSECTION_RELPATH)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--tile-size-px", type=int, default=896)
    parser.add_argument("--overlap-px", type=int, default=232)
    parser.add_argument("--keep-margin-px", type=int, default=116)
    parser.add_argument("--review-crop-pad-px", type=int, default=64)
    parser.add_argument("--tile-min-mask-ratio", type=float, default=0.02)
    parser.add_argument("--tile-min-mask-pixels", type=int, default=256)
    parser.add_argument("--tile-max-per-sample", type=int, default=0)
    parser.add_argument("--search-within-review-bbox", action="store_true")
    parser.add_argument("--fallback-to-all-if-empty", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_manifest = Path(args.output_manifest).resolve()
    split_roots = {}
    if str(args.train_root).strip():
        split_roots["train"] = Path(str(args.train_root).strip()).resolve()
    if str(args.val_root).strip():
        split_roots["val"] = Path(str(args.val_root).strip()).resolve()
    families = build_manifest_for_dataset(
        dataset_root=Path(args.dataset_root).resolve(),
        splits=[str(x) for x in args.splits],
        image_relpath=str(args.image_relpath),
        mask_relpath=str(args.mask_relpath),
        lane_relpath=str(args.lane_relpath),
        intersection_relpath=str(args.intersection_relpath),
        mask_threshold=int(args.mask_threshold),
        tile_size_px=int(args.tile_size_px),
        overlap_px=int(args.overlap_px),
        keep_margin_px=int(args.keep_margin_px),
        review_crop_pad_px=int(args.review_crop_pad_px),
        tile_min_mask_ratio=float(args.tile_min_mask_ratio),
        tile_min_mask_pixels=int(args.tile_min_mask_pixels),
        tile_max_per_sample=int(args.tile_max_per_sample),
        search_within_review_bbox=bool(args.search_within_review_bbox),
        fallback_to_all_if_empty=bool(args.fallback_to_all_if_empty),
        max_samples_per_split=int(args.max_samples_per_split),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        split_roots=split_roots or None,
    )
    count = write_jsonl(output_manifest, families)
    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_manifest": str(output_manifest),
        "splits": [str(x) for x in args.splits],
        "tile_size_px": int(args.tile_size_px),
        "overlap_px": int(args.overlap_px),
        "keep_margin_px": int(args.keep_margin_px),
        "review_crop_pad_px": int(args.review_crop_pad_px),
        "family_count": int(count),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
    }
    with output_manifest.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Built {count} families")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
