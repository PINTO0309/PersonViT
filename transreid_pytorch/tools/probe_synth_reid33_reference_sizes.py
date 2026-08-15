#!/usr/bin/env python3
"""Measure GPT Image 2 reference-size acceptance, quality, tokens, and cost.

The probe sends exactly one Low-quality Image Edit request per declared
reference-size variant.  It is deliberately separate from the production
pipeline: probe outputs can never be included in the 20,000-image dataset.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

from generate_synth_reid33 import PipelineError, _onnx_embeddings
from synth_reid33_core import (
    atomic_write_json,
    atomic_write_jsonl,
    load_config,
    make_cameras,
    process_image,
    read_jsonl,
    sha256_file,
    utc_now,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "configs" / "synth_reid33.yml"
DEFAULT_SOURCE_ROOT = HERE.parent / "data" / "SyntheticReID33_pilot_gpt_image_2"
DEFAULT_ROOT = HERE.parent / "data" / "SyntheticReID33_reference_size_probe"
MODEL = "gpt-image-2"
OUTPUT_SIZE = "576x1152"
QUALITY = "low"
REQUEST_VERSION = "reference-size-probe-v2-no-response-format"
ANCHOR_NAME = "anchor-p000-front.jpg"
PLATE_NAME = "plate-c00.jpg"

# Native variants preserve the source aspect ratio.  Fit variants exercise
# the compact 1:2 proxies considered for the production pipeline.
REFERENCE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "original",
        "anchor_size": (1024, 1536),
        "plate_size": (576, 1152),
        "anchor_mode": "resize",
    },
    {
        "name": "native_half",
        "anchor_size": (512, 768),
        "plate_size": (288, 576),
        "anchor_mode": "resize",
    },
    {
        "name": "native_quarter",
        "anchor_size": (256, 384),
        "plate_size": (144, 288),
        "anchor_mode": "resize",
    },
    {
        "name": "target_288x576_192x384",
        "anchor_size": (288, 576),
        "plate_size": (192, 384),
        "anchor_mode": "fit",
    },
    {
        "name": "uniform_256x512",
        "anchor_size": (256, 512),
        "plate_size": (256, 512),
        "anchor_mode": "fit",
    },
    {
        "name": "aggressive_192x384_128x256",
        "anchor_size": (192, 384),
        "plate_size": (128, 256),
        "anchor_mode": "fit",
    },
    {
        "name": "very_low_128x256_64x128",
        "anchor_size": (128, 256),
        "plate_size": (64, 128),
        "anchor_mode": "fit",
    },
)

PROMPT = (
    "The first input image is the exact fictional adult identity and wardrobe reference. "
    "The second input image is the exact empty fixed-camera background. Place that same adult "
    "as one full-body pedestrian in the background, walking naturally toward the camera with a "
    "slight right three-quarter body angle. Preserve the face, apparent age, auburn short hair, "
    "slender body build, muted green crew-neck shirt, beige chinos, black sneakers, and plain "
    "black briefcase. Keep the background camera position, downward perspective, architecture, "
    "and lighting unchanged. Show the complete head and both feet. One principal person only. "
    "No other people, readable text, logos, watermark, border, or collage."
)

STANDARD_PRICING = {
    "text_input": 5.0,
    "cached_text_input": 1.25,
    "image_input": 8.0,
    "cached_image_input": 2.0,
    "image_output": 30.0,
}


def _paths(root: Path) -> dict[str, Path]:
    return {
        "proxies": root / "references",
        "raw": root / "outputs" / "raw",
        "crops": root / "outputs" / "reid_crops",
        "results": root / "state" / "results.jsonl",
        "plan": root / "state" / "plan.json",
        "report": root / "report.json",
        "sheet": root / "contact_sheet.jpg",
    }


def _resize_reference(source: Path, destination: Path, size: tuple[int, int], mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if mode == "fit":
            resized = ImageOps.fit(
                image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )
        elif mode == "resize":
            resized = image.resize(size, Image.Resampling.LANCZOS)
        else:  # pragma: no cover - constant-table guard
            raise ValueError(f"unknown resize mode: {mode}")
        resized.save(destination, format="JPEG", quality=95, subsampling=0)


def prepare_references(source_root: Path, root: Path) -> list[dict[str, Any]]:
    paths = _paths(root)
    anchor_source = source_root / "assets" / ANCHOR_NAME
    plate_source = source_root / "assets" / PLATE_NAME
    for source in (anchor_source, plate_source):
        if not source.is_file():
            raise PipelineError(f"probe source is missing: {source}")
    rows = []
    for variant in REFERENCE_VARIANTS:
        directory = paths["proxies"] / str(variant["name"])
        anchor_path = directory / "anchor.jpg"
        plate_path = directory / "plate.jpg"
        _resize_reference(
            anchor_source,
            anchor_path,
            tuple(variant["anchor_size"]),
            str(variant["anchor_mode"]),
        )
        _resize_reference(
            plate_source,
            plate_path,
            tuple(variant["plate_size"]),
            "resize",
        )
        rows.append({
            **variant,
            "anchor_path": str(anchor_path.relative_to(root)),
            "plate_path": str(plate_path.relative_to(root)),
            "anchor_sha256": sha256_file(anchor_path),
            "plate_sha256": sha256_file(plate_path),
        })
    plan = {
        "created_at": utc_now(),
        "model": MODEL,
        "request_version": REQUEST_VERSION,
        "quality": QUALITY,
        "output_size": OUTPUT_SIZE,
        "requests": len(rows),
        "source_root": str(source_root.resolve()),
        "source_anchor": str(anchor_source.relative_to(source_root)),
        "source_plate": str(plate_source.relative_to(source_root)),
        "prompt": PROMPT,
        "variants": rows,
    }
    atomic_write_json(paths["plan"], plan)
    return rows


def _extract_usage(payload: Mapping[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    details = usage.get("input_tokens_details") or usage.get("input_tokens_detail") or {}
    text = int(details.get("text_tokens", 0) or 0)
    image = int(details.get("image_tokens", 0) or 0)
    return {
        "text_input_tokens": text,
        "cached_text_input_tokens": int(details.get("cached_text_tokens", 0) or 0),
        "image_input_tokens": image,
        "cached_image_input_tokens": int(details.get("cached_image_tokens", 0) or 0),
        "input_tokens_unclassified": max(0, int(usage.get("input_tokens", 0) or 0) - text - image),
        "image_output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def _usage_cost(usage: Mapping[str, int], prices: Mapping[str, float]) -> float:
    total = (
        int(usage.get("text_input_tokens", 0)) * float(prices["text_input"])
        + int(usage.get("cached_text_input_tokens", 0))
        * float(prices["cached_text_input"])
        + (
            int(usage.get("image_input_tokens", 0))
            + int(usage.get("input_tokens_unclassified", 0))
        )
        * float(prices["image_input"])
        + int(usage.get("cached_image_input_tokens", 0))
        * float(prices["cached_image_input"])
        + int(usage.get("image_output_tokens", 0)) * float(prices["image_output"])
    ) / 1_000_000.0
    return round(total, 8)


def _batch_prices(config: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in config["pricing_usd_per_million_tokens"]["batch"].items()
    }


def _result_error(variant: Mapping[str, Any], exc: Exception) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping):
        body = {}
    return {
        "variant": variant["name"],
        "anchor_size": list(variant["anchor_size"]),
        "plate_size": list(variant["plate_size"]),
        "status": "rejected",
        "request_version": REQUEST_VERSION,
        "attempted_at": utc_now(),
        "error": {
            "type": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
            "code": getattr(exc, "code", None) or body.get("code"),
            "message": str(exc),
            "request_id": getattr(exc, "request_id", None),
        },
    }


def _request_image(client: Any, anchor_path: Path, plate_path: Path) -> Any:
    """Send one GPT Image 2 edit using the endpoint's implicit base64 response."""
    with ExitStack() as stack:
        inputs = [
            stack.enter_context(anchor_path.open("rb")),
            stack.enter_context(plate_path.open("rb")),
        ]
        return client.images.edit(
            model=MODEL,
            image=inputs,
            prompt=PROMPT,
            n=1,
            size=OUTPUT_SIZE,
            quality=QUALITY,
            output_format="jpeg",
            output_compression=92,
            timeout=300.0,
        )


def run_probe(
    source_root: Path,
    root: Path,
    config: Mapping[str, Any],
    max_usd: float,
    retry_rejected: bool = False,
) -> list[dict[str, Any]]:
    if not math.isfinite(max_usd) or max_usd <= 0:
        raise ValueError("--max-usd must be a positive finite number")
    variants = prepare_references(source_root, root)
    if not os.environ.get("OPENAI_API_KEY"):
        raise PipelineError("OPENAI_API_KEY is not set; references were prepared but no API request was made")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise PipelineError("install the synth optional dependencies first") from exc

    paths = _paths(root)
    existing = {row["variant"]: row for row in read_jsonl(paths["results"])}
    pending = [
        variant
        for variant in variants
        if variant["name"] not in existing
        or existing[variant["name"]].get("request_version") != REQUEST_VERSION
        or (retry_rejected and existing[variant["name"]].get("status") == "rejected")
    ]
    already_spent = sum(
        float(row.get("standard_cost_usd", 0)) for row in existing.values()
    )
    conservative_per_request = 0.025
    if already_spent + conservative_per_request * len(pending) > max_usd:
        raise PipelineError(
            "cost stop: pending reference-size requests could exceed "
            f"${max_usd:.2f} using the ${conservative_per_request:.3f}/request reserve"
        )

    client = OpenAI()
    model = client.models.retrieve(MODEL)
    if str(getattr(model, "id", "")) != MODEL:
        raise PipelineError(f"model lookup did not return {MODEL}")
    batch_prices = _batch_prices(config)
    paths["raw"].mkdir(parents=True, exist_ok=True)
    for index, variant in enumerate(pending, 1):
        name = str(variant["name"])
        anchor_path = root / str(variant["anchor_path"])
        plate_path = root / str(variant["plate_path"])
        print(f"reference-size probe {index}/{len(pending)}: {name}", flush=True)
        try:
            response = _request_image(client, anchor_path, plate_path)
            payload = response.model_dump(mode="json")
            encoded = (payload.get("data") or [{}])[0].get("b64_json")
            if not encoded:
                raise PipelineError("successful image response contained no b64_json")
            raw_path = paths["raw"] / f"{name}.jpg"
            raw_path.write_bytes(base64.b64decode(encoded))
            with Image.open(raw_path) as output:
                decoded_size = list(output.size)
                decoded_format = output.format
                output.verify()
            usage = _extract_usage(payload)
            row = {
                "variant": name,
                "anchor_size": list(variant["anchor_size"]),
                "plate_size": list(variant["plate_size"]),
                "anchor_mode": variant["anchor_mode"],
                "status": "accepted",
                "request_version": REQUEST_VERSION,
                "attempted_at": utc_now(),
                "request_id": getattr(response, "_request_id", None),
                "model": MODEL,
                "quality": QUALITY,
                "output_size_requested": OUTPUT_SIZE,
                "output_path": str(raw_path.relative_to(root)),
                "output_size_decoded": decoded_size,
                "output_format_decoded": decoded_format,
                "output_sha256": sha256_file(raw_path),
                "usage": usage,
                "standard_cost_usd": _usage_cost(usage, STANDARD_PRICING),
                "batch_equivalent_cost_usd": _usage_cost(usage, batch_prices),
            }
        except Exception as exc:  # one attempt per resolution is intentional
            row = _result_error(variant, exc)
        existing[name] = row
        atomic_write_jsonl(
            paths["results"],
            [existing[key] for key in sorted(existing)],
        )
        if row.get("error", {}).get("code") in {
            "unknown_parameter",
            "invalid_api_key",
            "model_not_found",
        }:
            print(
                "stopping after a request-wide API error; remaining variants were not sent",
                file=sys.stderr,
            )
            break
    return [existing[key] for key in sorted(existing)]


def _write_contact_sheet(root: Path, variants: Sequence[Mapping[str, Any]]) -> Path:
    paths = _paths(root)
    results = {row["variant"]: row for row in read_jsonl(paths["results"])}
    cell_w, cell_h, label_h = 180, 300, 42
    headers = ("identity ref", "camera ref", "API output", "128x256 crop")
    sheet = Image.new(
        "RGB",
        (len(headers) * cell_w, (len(variants) + 1) * (cell_h + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, header in enumerate(headers):
        draw.text((column * cell_w + 4, cell_h + 8), header, fill="black")
    for row_index, variant in enumerate(variants, 1):
        result = results.get(str(variant["name"]), {})
        image_paths = [
            root / str(variant["anchor_path"]),
            root / str(variant["plate_path"]),
            root / str(result["output_path"]) if result.get("output_path") else None,
            paths["crops"] / f"{variant['name']}.jpg",
        ]
        for column, image_path in enumerate(image_paths):
            if image_path is None or not image_path.is_file():
                continue
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                image.thumbnail((cell_w, cell_h))
            x = column * cell_w + (cell_w - image.width) // 2
            y = row_index * (cell_h + label_h) + (cell_h - image.height) // 2
            sheet.paste(image, (x, y))
        usage = result.get("usage") or {}
        label = (
            f"{variant['name']}  imgTok={usage.get('image_input_tokens', '-')}  "
            f"std=${float(result.get('standard_cost_usd', 0)):.4f}"
        )
        draw.text((4, row_index * (cell_h + label_h) + cell_h + 8), label, fill="black")
    paths["sheet"].parent.mkdir(parents=True, exist_ok=True)
    sheet.save(paths["sheet"], quality=92)
    return paths["sheet"]


def build_report(
    source_root: Path, root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    variants = prepare_references(source_root, root)
    paths = _paths(root)
    results = {row["variant"]: row for row in read_jsonl(paths["results"])}
    cameras = make_cameras(config)
    successful = []
    for variant in variants:
        result = results.get(str(variant["name"]))
        if not result or result.get("status") != "accepted":
            continue
        raw_path = root / str(result["output_path"])
        crop_path = paths["crops"] / f"{variant['name']}.jpg"
        qa = process_image(
            raw_path,
            crop_path,
            cameras[0],
            {"occluded": False, "generation_seed": 20260815},
            config,
        )
        result["qa"] = qa
        result["crop_path"] = str(crop_path.relative_to(root))
        successful.append(result)

    if successful:
        anchor_path = source_root / "assets" / ANCHOR_NAME
        crop_paths = [root / str(row["crop_path"]) for row in successful]
        for model_key, config_key in (("vit", "vit_onnx"), ("osnet", "osnet_onnx")):
            model_path = (DEFAULT_CONFIG.parent / config["qa"][config_key]).resolve()
            features = _onnx_embeddings(model_path, [anchor_path, *crop_paths])
            anchor = features[0]
            candidates = features[1:]
            baseline = candidates[0]
            for index, row in enumerate(successful):
                row.setdefault("qa", {}).setdefault("embedding", {})[model_key] = {
                    "anchor_cosine_similarity": round(float(candidates[index] @ anchor), 7),
                    "baseline_output_cosine_similarity": round(
                        float(candidates[index] @ baseline), 7
                    ),
                    "model_sha256": sha256_file(model_path),
                }
    atomic_write_jsonl(paths["results"], [results[key] for key in sorted(results)])
    sheet = _write_contact_sheet(root, variants)
    original = results.get("original") or {}
    original_tokens = int((original.get("usage") or {}).get("image_input_tokens", 0))
    report_rows = []
    for variant in variants:
        result = results.get(str(variant["name"]), {})
        image_tokens = int((result.get("usage") or {}).get("image_input_tokens", 0))
        batch_cost = result.get("batch_equivalent_cost_usd")
        report_rows.append({
            "variant": variant["name"],
            "anchor_size": list(variant["anchor_size"]),
            "plate_size": list(variant["plate_size"]),
            "status": result.get("status", "not_run"),
            "image_input_tokens": image_tokens or None,
            "image_input_token_reduction_vs_original": (
                round(1 - image_tokens / original_tokens, 6)
                if image_tokens and original_tokens
                else None
            ),
            "standard_cost_usd": result.get("standard_cost_usd"),
            "batch_equivalent_cost_usd": batch_cost,
            "projected_20000_sample_batch_cost_usd": (
                round(float(batch_cost) * 20_000, 2) if batch_cost is not None else None
            ),
            "geometry_pass": (result.get("qa") or {}).get("geometry_pass"),
            "qa_reason": (result.get("qa") or {}).get("reason"),
            "embedding": (result.get("qa") or {}).get("embedding"),
            "error": result.get("error"),
        })
    report = {
        "created_at": utc_now(),
        "model": MODEL,
        "request_version": REQUEST_VERSION,
        "quality": QUALITY,
        "output_size": OUTPUT_SIZE,
        "one_request_per_variant": True,
        "automatic_retries": 0,
        "standard_actual_cost_usd": round(
            sum(float(row.get("standard_cost_usd", 0)) for row in results.values()), 8
        ),
        "batch_equivalent_cost_usd": round(
            sum(float(row.get("batch_equivalent_cost_usd", 0)) for row in results.values()), 8
        ),
        "pricing_snapshot": {
            "standard": STANDARD_PRICING,
            "batch": _batch_prices(config),
        },
        "contact_sheet": str(sheet.relative_to(root)),
        "variants": report_rows,
    }
    atomic_write_json(paths["report"], report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="create all local reference proxies; no API call")
    run = subparsers.add_parser("run", help="send one synchronous Image Edit per resolution")
    run.add_argument("--max-usd", type=float, default=0.20)
    run.add_argument("--retry-rejected", action="store_true")
    subparsers.add_parser("report", help="run local QA and write the comparison report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        root = args.root.resolve()
        source_root = args.source_root.resolve()
        if args.command == "prepare":
            variants = prepare_references(source_root, root)
            print(f"prepared {len(variants)} variants under {root}; no API request was made")
        elif args.command == "run":
            results = run_probe(
                source_root,
                root,
                config,
                args.max_usd,
                retry_rejected=args.retry_rejected,
            )
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            report = build_report(source_root, root, config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (PipelineError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
