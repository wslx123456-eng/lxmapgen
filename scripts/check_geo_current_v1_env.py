from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from typing import Dict


def try_import_transformers_qwen() -> Dict[str, str]:
    try:
        import transformers  # type: ignore
    except Exception as exc:
        return {"ok": "false", "detail": f"import transformers failed: {exc}"}
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: F401
    except Exception as exc:
        return {
            "ok": "false",
            "detail": f"transformers={getattr(transformers, '__version__', 'unknown')} missing Qwen2.5-VL runtime support: {exc}",
        }
    return {"ok": "true", "detail": f"transformers={getattr(transformers, '__version__', 'unknown')}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight check for geo current v1 training environment.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-qwen25-vl", action="store_true")
    args = parser.parse_args()

    result: Dict[str, object] = {
        "python": sys.version.split()[0],
        "llamafactory_cli": shutil.which("llamafactory-cli") or "",
    }

    try:
        import torch

        result["torch"] = str(torch.__version__)
        result["torch_cuda"] = str(torch.version.cuda)
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        try:
            result["bf16_supported"] = bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
        except Exception:
            result["bf16_supported"] = False
    except Exception as exc:
        result["torch_import_error"] = str(exc)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    qwen_status = try_import_transformers_qwen()
    result["qwen25_vl_import_ok"] = qwen_status["ok"] == "true"
    result["qwen25_vl_import_detail"] = qwen_status["detail"]

    exit_code = 0
    if args.require_cuda and not bool(result.get("cuda_available")):
        exit_code = 3
    if args.require_qwen25_vl and not bool(result.get("qwen25_vl_import_ok")):
        exit_code = 4

    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
