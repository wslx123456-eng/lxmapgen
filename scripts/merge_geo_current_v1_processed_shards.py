import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def merge_jsonl_files(paths: List[Path], output_path: Path) -> Tuple[int, int]:
    seen = set()
    merged: List[Dict] = []
    duplicate_count = 0
    for path in paths:
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            row_id = row.get("id", None)
            key = ("id", str(row_id)) if row_id is not None else ("json", json.dumps(row, sort_keys=True, ensure_ascii=False))
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            merged.append(row)
    return write_jsonl(output_path, merged), duplicate_count


def copy_tree_files(src_root: Path, dst_root: Path) -> int:
    copied = 0
    if not src_root.is_dir():
        return copied
    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        ensure_dir(dst.parent)
        if dst.exists():
            if dst.stat().st_size != src.stat().st_size:
                raise RuntimeError(f"Conflicting file while merging shards: {dst}")
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def build_dataset_registry(output_root: Path, prefix: str) -> Dict[str, Dict]:
    registry: Dict[str, Dict] = {}
    for split in ("train", "val"):
        dataset_name = f"{prefix}_{split}"
        dataset_file = output_root / f"{split}.jsonl"
        if not dataset_file.is_file():
            continue
        registry[dataset_name] = {
            "file_name": str(dataset_file.resolve()),
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
    return registry


def merge_stage(shard_roots: List[Path], output_root: Path, stage_name: str, prefix: str) -> Dict:
    dataset_root = output_root / stage_name / "dataset"
    ensure_dir(dataset_root)
    copied_images = 0
    for shard_root in shard_roots:
        copied_images += copy_tree_files(shard_root / stage_name / "dataset" / "images", dataset_root / "images")
    split_summary: Dict[str, Dict] = {}
    for split in ("train", "val"):
        sample_count, dup_count = merge_jsonl_files(
            [root / stage_name / "dataset" / f"{split}.jsonl" for root in shard_roots],
            dataset_root / f"{split}.jsonl",
        )
        meta_count, meta_dup_count = merge_jsonl_files(
            [root / stage_name / "dataset" / f"meta_{split}.jsonl" for root in shard_roots],
            dataset_root / f"meta_{split}.jsonl",
        )
        split_summary[split] = {
            "samples": int(sample_count),
            "sample_duplicates_skipped": int(dup_count),
            "meta_samples": int(meta_count),
            "meta_duplicates_skipped": int(meta_dup_count),
        }
    dataset_registry = build_dataset_registry(dataset_root, prefix)
    with (dataset_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_registry, f, ensure_ascii=False, indent=2)
    summary = {
        "stage": stage_name,
        "dataset_name_prefix": prefix,
        "shard_roots": [str(path.resolve()) for path in shard_roots],
        "copied_images": int(copied_images),
        "summary": split_summary,
    }
    with (dataset_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def merge_manifest(shard_roots: List[Path], output_root: Path) -> Dict:
    manifest_out = output_root / "family_manifest.jsonl"
    count, dup_count = merge_jsonl_files(
        [root / "family_manifest.jsonl" for root in shard_roots],
        manifest_out,
    )
    summary = {
        "family_count": int(count),
        "duplicates_skipped": int(dup_count),
        "shard_roots": [str(path.resolve()) for path in shard_roots],
    }
    with (output_root / "family_manifest.summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multi-machine processed geo-current v1 shard outputs.")
    parser.add_argument("--shard-roots", type=str, nargs="+", required=True)
    parser.add_argument("--output-root", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_roots = [Path(path).resolve() for path in args.shard_roots]
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)
    manifest_summary = merge_manifest(shard_roots, output_root)
    stage_a_summary = merge_stage(shard_roots, output_root, "stage_a", "unimapgen_geo_current_patch_only")
    stage_b_summary = merge_stage(shard_roots, output_root, "stage_b", "unimapgen_geo_current_state")
    summary = {
        "output_root": str(output_root),
        "manifest": manifest_summary,
        "stage_a": stage_a_summary,
        "stage_b": stage_b_summary,
    }
    with (output_root / "merge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(shard_roots)} shard roots into {output_root}")


if __name__ == "__main__":
    main()
