import argparse
import json
from pathlib import Path
from typing import Dict, List

from geo_current_dataset_v1_common import (
    DEFAULT_STAGEA_PROMPT_TEMPLATE,
    DEFAULT_STAGEA_SYSTEM_PROMPT,
    build_full_segments_for_patch,
    build_patch_image,
    build_patch_only_record,
    build_patch_target_lines,
    build_patch_target_lines_quantized,
    build_patch_target_lines_float,
    ensure_dir,
    family_global_lines,
    load_family_raster_and_mask,
    load_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export patch-only LLaMAFactory SFT data from current-dataset family manifests.")
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--resample-step-px", type=float, default=4.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--include-lane", action="store_true")
    parser.add_argument("--include-intersection-boundary", action="store_true")
    parser.add_argument("--max-families-per-split", type=int, default=0)
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_STAGEA_SYSTEM_PROMPT)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_STAGEA_PROMPT_TEMPLATE)
    return parser.parse_args()


def export_split(
    split: str,
    families: List[Dict],
    output_root: Path,
    band_indices,
    mask_threshold: int,
    resample_step_px: float,
    boundary_tol_px: float,
    include_lane: bool,
    include_intersection_boundary: bool,
    max_families_per_split: int,
    system_prompt: str,
    prompt_template: str,
) -> Dict[str, int]:
    rows: List[Dict] = []
    meta_rows: List[Dict] = []
    family_count = 0
    for family in families:
        if str(family.get("split")) != str(split):
            continue
        family_count += 1
        if int(max_families_per_split) > 0 and family_count > int(max_families_per_split):
            break
        raw_image_hwc, raster_meta, review_mask = load_family_raster_and_mask(
            family=family,
            band_indices=band_indices,
            mask_threshold=int(mask_threshold),
        )
        global_lines = family_global_lines(
            family=family,
            raster_meta=raster_meta,
            include_lane=bool(include_lane),
            include_intersection=bool(include_intersection_boundary),
        )
        for patch in sorted(list(family["patches"]), key=lambda item: int(item["patch_id"])):
            patch_id = int(patch["patch_id"])
            patch_image = build_patch_image(raw_image_hwc=raw_image_hwc, patch=patch)
            full_segments = build_full_segments_for_patch(
                patch=patch,
                global_lines=global_lines,
                review_mask=review_mask,
                resample_step_px=float(resample_step_px),
                boundary_tol_px=float(boundary_tol_px),
            )
            target_lines = build_patch_target_lines(full_segments, patch=patch)
            target_lines_quantized = build_patch_target_lines_quantized(full_segments, patch=patch)
            target_lines_float = build_patch_target_lines_float(full_segments, patch=patch)
            image_rel = Path("images") / split / str(family["family_id"]) / f"p{patch_id:04d}.png"
            out_image = output_root / image_rel
            ensure_dir(out_image.parent)
            patch_image.save(out_image)
            sample_id = f"{family['family_id']}_p{patch_id:04d}"
            rows.append(
                build_patch_only_record(
                    image_rel_path=image_rel.as_posix(),
                    target_lines=target_lines,
                    sample_id=sample_id,
                    system_prompt=system_prompt,
                    prompt_template=prompt_template,
                )
            )
            meta_rows.append(
                {
                    "id": sample_id,
                    "split": split,
                    "family_id": family["family_id"],
                    "source_sample_id": family.get("source_sample_id", ""),
                    "source_image": family["source_image"],
                    "source_image_path": family.get("source_image_path", ""),
                    "source_mask_path": family.get("source_mask_path", ""),
                    "source_lane_path": family.get("source_lane_path", ""),
                    "source_intersection_path": family.get("source_intersection_path", ""),
                    "image_size": family.get("image_size", []),
                    "patch_id": patch_id,
                    "row": int(patch["row"]),
                    "col": int(patch["col"]),
                    "crop_box": patch["crop_box"],
                    "keep_box": patch["keep_box"],
                    "mask_ratio": float(patch.get("mask_ratio", 0.0)),
                    "mask_pixels": int(patch.get("mask_pixels", 0)),
                    "num_target_lines": len(target_lines),
                    "target_lines": target_lines,
                    "target_lines_quantized": target_lines_quantized,
                    "target_lines_float": target_lines_float,
                }
            )
    count_main = write_jsonl(output_root / f"{split}.jsonl", rows)
    count_meta = write_jsonl(output_root / f"meta_{split}.jsonl", meta_rows)
    return {"families": family_count if int(max_families_per_split) <= 0 else min(family_count, int(max_families_per_split)), "samples": count_main, "meta_samples": count_meta}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    families = load_jsonl(Path(args.family_manifest).resolve())
    include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
    include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)
    system_prompt = str(args.system_prompt).strip() if bool(args.use_system_prompt) else ""
    summary: Dict[str, Dict[str, int]] = {}
    dataset_registry: Dict[str, Dict] = {}
    for split in args.splits:
        summary[str(split)] = export_split(
            split=str(split),
            families=families,
            output_root=output_root,
            band_indices=[int(x) for x in args.band_indices],
            mask_threshold=int(args.mask_threshold),
            resample_step_px=float(args.resample_step_px),
            boundary_tol_px=float(args.boundary_tol_px),
            include_lane=include_lane,
            include_intersection_boundary=include_intersection_boundary,
            max_families_per_split=int(args.max_families_per_split),
            system_prompt=system_prompt,
            prompt_template=str(args.prompt_template),
        )
        dataset_name = f"unimapgen_geo_current_patch_only_{str(split)}"
        dataset_registry[dataset_name] = {
            "file_name": str((output_root / f"{split}.jsonl").resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    with (output_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_registry, f, ensure_ascii=False, indent=2)
    export_summary = {
        "dataset_name_prefix": "unimapgen_geo_current_patch_only",
        "source_family_manifest": str(Path(args.family_manifest).resolve()),
        "prompt_template": str(args.prompt_template),
        "use_system_prompt": bool(system_prompt),
        "system_prompt": system_prompt,
        "summary": summary,
    }
    with (output_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(export_summary, f, ensure_ascii=False, indent=2)
    for split, info in summary.items():
        print(f"[{split}] families={info['families']} samples={info['samples']} meta={info['meta_samples']}")
    print(f"Saved dataset to {output_root}")


if __name__ == "__main__":
    main()
