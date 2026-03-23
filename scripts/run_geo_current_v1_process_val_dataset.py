import argparse
import json
from pathlib import Path
from typing import Dict, List

from geo_current_dataset_v1_common import (
    DEFAULT_STAGEA_SYSTEM_PROMPT,
    build_manifest_for_dataset,
    build_patch_image,
    ensure_dir,
    load_family_raster_and_mask,
    write_jsonl,
)


DEFAULT_INFER_USER_PROMPT = """<image>
Please predict the complete road-structure line map for the current satellite patch.
Basic patch info:
- family_id: {family_id}
- patch_id: {patch_id}
- crop_box: {crop_box}
- keep_box: {keep_box}

Return only valid JSON with schema {{"lines":[...]}}.
Use category lane_line for roads and intersection_polygon for intersections.
All points must be in patch-local UV coordinates of the current patch.
Use patch-local integer UV coordinates where one pixel equals one unit."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build val-only inference dataset with patch images and prompt-only records.")
    parser.add_argument("--dataset-root", type=str, default="/dataset/zsy/dataset-extracted")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--image-relpath", type=str, default="patch_tif/0.tif")
    parser.add_argument("--mask-relpath", type=str, default="patch_tif/0_edit_poly.tif")
    parser.add_argument("--lane-relpath", type=str, default="label_check_crop/Lane.geojson")
    parser.add_argument("--intersection-relpath", type=str, default="label_check_crop/Intersection.geojson")
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
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_STAGEA_SYSTEM_PROMPT)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_INFER_USER_PROMPT)
    return parser.parse_args()


def build_infer_record(image_rel_path: str, sample_id: str, system_prompt: str, prompt_text: str) -> Dict:
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_text)})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    manifest_path = output_root / "family_manifest_val.jsonl"
    images_root = output_root / "images" / "val"
    infer_jsonl = output_root / "infer_val.jsonl"
    meta_jsonl = output_root / "meta_val.jsonl"

    families = build_manifest_for_dataset(
        dataset_root=Path(args.dataset_root).resolve(),
        splits=["val"],
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
        max_samples_per_split=int(args.max_samples),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )

    write_jsonl(manifest_path, families)

    infer_rows: List[Dict] = []
    meta_rows: List[Dict] = []
    system_prompt = str(args.system_prompt).strip() if bool(args.use_system_prompt) else ""
    for family in families:
        raw_image_hwc, _, _ = load_family_raster_and_mask(
            family=family,
            band_indices=[int(x) for x in args.band_indices],
            mask_threshold=int(args.mask_threshold),
        )
        family_id = str(family["family_id"])
        for patch in sorted(list(family["patches"]), key=lambda item: int(item["patch_id"])):
            patch_id = int(patch["patch_id"])
            patch_image = build_patch_image(raw_image_hwc=raw_image_hwc, patch=patch)
            image_rel = Path("images") / "val" / family_id / f"p{patch_id:04d}.png"
            out_image = output_root / image_rel
            ensure_dir(out_image.parent)
            patch_image.save(out_image)

            sample_id = f"{family_id}_p{patch_id:04d}"
            prompt_text = str(args.prompt_template).format(
                family_id=family_id,
                patch_id=patch_id,
                crop_box=json.dumps(patch["crop_box"], ensure_ascii=False, separators=(",", ":")),
                keep_box=json.dumps(patch["keep_box"], ensure_ascii=False, separators=(",", ":")),
            )
            infer_rows.append(
                build_infer_record(
                    image_rel_path=image_rel.as_posix(),
                    sample_id=sample_id,
                    system_prompt=system_prompt,
                    prompt_text=prompt_text,
                )
            )
            meta_rows.append(
                {
                    "id": sample_id,
                    "split": "val",
                    "family_id": family_id,
                    "source_image": family["source_image"],
                    "patch_id": patch_id,
                    "row": int(patch["row"]),
                    "col": int(patch["col"]),
                    "crop_box": patch["crop_box"],
                    "keep_box": patch["keep_box"],
                    "mask_ratio": float(patch.get("mask_ratio", 0.0)),
                    "mask_pixels": int(patch.get("mask_pixels", 0)),
                    "prompt_text": prompt_text,
                }
            )

    infer_count = write_jsonl(infer_jsonl, infer_rows)
    meta_count = write_jsonl(meta_jsonl, meta_rows)
    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "infer_jsonl": str(infer_jsonl),
        "meta_jsonl": str(meta_jsonl),
        "split": "val",
        "family_count": len(families),
        "sample_count": int(infer_count),
        "meta_count": int(meta_count),
        "tile_size_px": int(args.tile_size_px),
        "overlap_px": int(args.overlap_px),
        "keep_margin_px": int(args.keep_margin_px),
        "review_crop_pad_px": int(args.review_crop_pad_px),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "use_system_prompt": bool(system_prompt),
    }
    with (output_root / "infer_val.summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[GeoCurrentV1] built val inference dataset families={len(families)} samples={infer_count}")
    print(f"[GeoCurrentV1] output_root={output_root}")


if __name__ == "__main__":
    main()
