import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_llamafactory_patch_only_from_raw_family_manifest import PATCH_ONLY_PROMPT_TEMPLATE, PATCH_ONLY_SYSTEM_PROMPT
from export_llamafactory_state_sft_from_raw_family_manifest import (
    DEFAULT_PROMPT_TEMPLATE as STATE_PROMPT_TEMPLATE,
    build_owned_segments_global,
    build_patch_ownership_rect_global,
    build_target_lines_for_patch,
    clamp_points,
    collect_global_lines,
    dedup_points,
    extract_state_lines,
    simplify_for_json,
    sort_lines,
)
from run_qwen2_5vl_lora_small_eval import parse_generated_json, render_panel, stack_panels
from unimapgen.compare_metrics import _sample_metrics

STATE_SYSTEM_PROMPT = (
    "You are a road-map reconstruction assistant for satellite-image patches.\n\n"
    "Your task is to predict the road map of the current image patch from:\n"
    "1. the satellite image patch, and\n"
    "2. the previous map state provided in the prompt.\n\n"
    "The previous state contains road segments that were cut from already processed neighboring patches.\n"
    "You must use these cut traces to maintain cross-patch continuity whenever they enter the current patch.\n\n"
    "Requirements:\n"
    "- Predict the road map for the current patch.\n"
    "- Continue incoming cut traces when appropriate.\n"
    "- Preserve geometric continuity across patch boundaries.\n"
    "- Do not add explanations, commentary, or extra text.\n"
    "- Return only valid JSON in the required schema.\n"
    "- Do not output markdown fences.\n"
    "- Do not omit required fields.\n"
    "- Keep all coordinates in the patch-local coordinate system used by the sample.\n\n"
    "If no previous state is provided, start from the current patch image only.\n"
    "If a road segment is truncated by the current patch boundary, mark the endpoint consistently with the schema."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout inference on raw paper16 families.")
    parser.add_argument("--ann-json", type=str, required=True)
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-families", type=int, default=16)
    parser.add_argument("--family-ids", type=str, default="")
    parser.add_argument("--accepted-categories", type=str, nargs="+", default=["lane_line", "virtual_line", "curb"])
    parser.add_argument("--output-category", type=str, default="road")
    parser.add_argument("--resample-step-px", type=float, default=12.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--trace-points", type=int, default=8)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--engine", type=str, default="custom", choices=["custom", "llamafactory"])
    parser.add_argument("--template", type=str, default="qwen2_vl")
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--use-patch-only-prompt-when-empty", action="store_true")
    parser.add_argument("--export-visualizations", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt_and_system(state_lines: Sequence[Dict], use_patch_only_prompt_when_empty: bool) -> Tuple[str, str]:
    if len(state_lines) == 0 and bool(use_patch_only_prompt_when_empty):
        return PATCH_ONLY_PROMPT_TEMPLATE, PATCH_ONLY_SYSTEM_PROMPT
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    return STATE_PROMPT_TEMPLATE.format(state_json=state_json), STATE_SYSTEM_PROMPT


def build_conversation(prompt_text: str, system_text: str, image_ref: str) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = []
    if str(system_text).strip():
        conversation.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    user_text = prompt_text.replace("<image>\n", "", 1).replace("<image>", "", 1).strip()
    conversation.append(
        {
            "role": "user",
            "content": [{"type": "image", "image": image_ref}, {"type": "text", "text": user_text}],
        }
    )
    return conversation


def generate_with_custom_engine(
    prompt_text: str,
    system_text: str,
    image: Image.Image,
    image_ref: str,
    processor: AutoProcessor,
    model: torch.nn.Module,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    conversation = build_conversation(prompt_text, system_text, image_ref)
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            use_cache=True,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = generated[:, prompt_len:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def sanitize_pred_lines(pred_lines: Sequence[Dict], patch_size: int, output_category: str) -> List[Dict]:
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        points = simplify_for_json(arr, patch_size=patch_size)
        if len(points) < 2:
            continue
        out.append(
            {
                "category": str(line.get("category", output_category)),
                "start_type": str(line.get("start_type", "start")),
                "end_type": str(line.get("end_type", "end")),
                "points": points,
            }
        )
    return sort_lines(out)


def predicted_lines_to_global_arrays(pred_lines: Sequence[Dict], patch: Dict) -> List[np.ndarray]:
    crop_box = patch["crop_box"]
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    out: List[np.ndarray] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        out.append(dedup_points(arr + offset))
    return out


def main() -> None:
    args = parse_args()
    annotations = load_json(Path(args.ann_json).resolve())
    families = load_jsonl(Path(args.family_manifest).resolve())
    output_root = Path(args.output_root).resolve()
    predictions_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    viz_dir = output_root / "visualizations"
    for path in [output_root, predictions_dir, metrics_dir, viz_dir]:
        ensure_dir(path)

    wanted_family_ids = {x.strip() for x in str(args.family_ids).split(",") if x.strip()}
    selected_families: List[Dict] = []
    for family in families:
        if str(family.get("split")) != str(args.split):
            continue
        if wanted_family_ids and str(family.get("family_id")) not in wanted_family_ids:
            continue
        selected_families.append(family)
        if int(args.max_families) > 0 and len(selected_families) >= int(args.max_families):
            break

    processor = None
    model = None
    chat_model = None
    if args.engine == "custom":
        processor_path = args.processor_path or args.adapter
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if args.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        if not args.device.startswith("cuda"):
            model = model.to(args.device)
        model.eval()
    else:
        from llamafactory.chat import ChatModel

        infer_args = {
            "model_name_or_path": args.base_model,
            "adapter_name_or_path": args.adapter,
            "finetuning_type": "lora",
            "stage": "sft",
            "template": args.template,
            "infer_backend": "huggingface",
            "infer_dtype": "bfloat16",
            "trust_remote_code": True,
            "image_max_pixels": int(args.image_max_pixels),
        }
        chat_model = ChatModel(infer_args)

    overall_results: List[Dict[str, Any]] = []
    overall_metrics: Dict[str, List[float]] = {}
    for family in selected_families:
        family_id = str(family["family_id"])
        ann = annotations.get(str(family["source_image"]))
        if not isinstance(ann, dict):
            continue
        patches = sorted(list(family["patches"]), key=lambda x: int(x["patch_id"]))
        grid_size = int(round(math.sqrt(len(patches))))
        global_lines = collect_global_lines(ann=ann, accepted_categories=[str(x) for x in args.accepted_categories])

        ownership_rects_by_patch: Dict[int, Tuple[float, float, float, float]] = {}
        gt_owned_segments_by_patch: Dict[int, List[Dict]] = {}
        for patch in patches:
            patch_id = int(patch["patch_id"])
            ownership_rect = build_patch_ownership_rect_global(patches=patches, patch_id=patch_id, grid_size=grid_size)
            ownership_rects_by_patch[patch_id] = ownership_rect
            gt_owned_segments_by_patch[patch_id] = build_owned_segments_global(
                global_lines=global_lines,
                ownership_rect_global=ownership_rect,
                resample_step_px=float(args.resample_step_px),
                boundary_tol_px=float(args.boundary_tol_px),
            )

        pred_owned_segments_by_patch: Dict[int, List[Dict]] = {}
        family_records: List[Dict[str, Any]] = []
        family_metric_lists: Dict[str, List[float]] = {}
        with Image.open(str(family["source_image_path"])) as raw_img:
            raw_image = raw_img.convert("RGB")
            for patch in patches:
                patch_id = int(patch["patch_id"])
                crop_box = patch["crop_box"]
                patch_image = raw_image.crop(
                    (int(crop_box["x_min"]), int(crop_box["y_min"]), int(crop_box["x_max"]), int(crop_box["y_max"]))
                )
                patch_size = int(crop_box["x_max"] - crop_box["x_min"])
                ownership_rect_global = ownership_rects_by_patch[patch_id]
                ownership_rect_local = (
                    float(ownership_rect_global[0] - crop_box["x_min"]),
                    float(ownership_rect_global[1] - crop_box["y_min"]),
                    float(ownership_rect_global[2] - crop_box["x_min"]),
                    float(ownership_rect_global[3] - crop_box["y_min"]),
                )
                gt_lines = build_target_lines_for_patch(
                    owned_segments_global=gt_owned_segments_by_patch[patch_id],
                    patch=patch,
                    output_category=str(args.output_category),
                )
                state_lines = extract_state_lines(
                    patch_id=patch_id,
                    patches=patches,
                    grid_size=grid_size,
                    ownership_rect_global=ownership_rect_global,
                    owned_segments_by_patch=pred_owned_segments_by_patch,
                    trace_points=int(args.trace_points),
                    output_category=str(args.output_category),
                    boundary_tol_px=float(args.boundary_tol_px),
                )
                prompt_text, system_text = build_prompt_and_system(
                    state_lines=state_lines,
                    use_patch_only_prompt_when_empty=bool(args.use_patch_only_prompt_when_empty),
                )
                image_ref = f"{family_id}_p{patch_id:02d}.png"
                if args.engine == "custom":
                    pred_text = generate_with_custom_engine(
                        prompt_text=prompt_text,
                        system_text=system_text,
                        image=patch_image,
                        image_ref=image_ref,
                        processor=processor,
                        model=model,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                    )
                else:
                    user_text = prompt_text.replace("<image>\n", "", 1).replace("<image>", "", 1).strip()
                    responses = chat_model.chat(
                        messages=[{"role": "user", "content": user_text}],
                        system=system_text,
                        images=[patch_image],
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    pred_text = responses[0].response_text if responses else ""

                pred_obj, cleaned_pred = parse_generated_json(pred_text)
                pred_lines = sanitize_pred_lines(list((pred_obj or {"lines": []}).get("lines", [])), patch_size, str(args.output_category))
                pred_owned_segments_by_patch[patch_id] = build_owned_segments_global(
                    global_lines=predicted_lines_to_global_arrays(pred_lines, patch),
                    ownership_rect_global=ownership_rect_global,
                    resample_step_px=float(args.resample_step_px),
                    boundary_tol_px=float(args.boundary_tol_px),
                )
                metrics = _sample_metrics(pred_lines=pred_lines, gt_lines=gt_lines, thresholds=[2.0, 4.0, 8.0])
                for key, value in metrics.items():
                    family_metric_lists.setdefault(key, []).append(float(value))
                    overall_metrics.setdefault(key, []).append(float(value))

                if bool(args.export_visualizations):
                    family_viz_dir = viz_dir / family_id
                    ensure_dir(family_viz_dir)
                    gt_panel = render_panel(
                        image=patch_image,
                        lines=gt_lines,
                        state_lines=state_lines,
                        title=f"{family_id}_p{patch_id:02d} | GT",
                        line_color=(0, 120, 255),
                        size=896,
                    )
                    pred_panel = render_panel(
                        image=patch_image,
                        lines=pred_lines,
                        state_lines=state_lines,
                        title=f"{family_id}_p{patch_id:02d} | Pred",
                        line_color=(255, 60, 60),
                        size=896,
                    )
                    for panel in [gt_panel, pred_panel]:
                        d = ImageDraw.Draw(panel)
                        d.rectangle(ownership_rect_local, outline=(255, 215, 0), width=2)
                    stack_panels(gt_panel, pred_panel).save(family_viz_dir / f"p{patch_id:02d}.png")

                family_records.append(
                    {
                        "patch_id": patch_id,
                        "prompt_state_lines": state_lines,
                        "neighbor_sources": sorted({int(x["source_patch"]) for x in state_lines}),
                        "pred_text": pred_text,
                        "pred_json_text": cleaned_pred,
                        "pred_lines": pred_lines,
                        "gt_lines": gt_lines,
                        "parse_ok": pred_obj is not None,
                        "metrics": metrics,
                    }
                )

        family_summary = {
            "family_id": family_id,
            "num_patches": len(family_records),
            "num_parse_ok": sum(1 for rec in family_records if rec["parse_ok"]),
            "mean_metrics": {k: sum(v) / max(1, len(v)) for k, v in family_metric_lists.items()},
        }
        with (predictions_dir / f"{family_id}.json").open("w", encoding="utf-8") as f:
            json.dump({"family_id": family_id, "patches": family_records}, f, ensure_ascii=False, indent=2)
        with (metrics_dir / f"{family_id}.json").open("w", encoding="utf-8") as f:
            json.dump(family_summary, f, ensure_ascii=False, indent=2)
        overall_results.append(family_summary)

    overall_summary = {
        "num_families": len(overall_results),
        "num_patches": int(sum(item["num_patches"] for item in overall_results)),
        "num_parse_ok": int(sum(item["num_parse_ok"] for item in overall_results)),
        "mean_metrics": {k: sum(v) / max(1, len(v)) for k, v in overall_metrics.items()},
        "families": overall_results,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
