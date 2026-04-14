#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUCCESS_LABEL = "good_tracking"
REVIEW_LABELS = {
    SUCCESS_LABEL,
    "partial_tracking",
    "drift",
    "wrong_object",
    "no_detection",
    "fallback",
}
LABEL_RANK = {
    "good_tracking": 3,
    "partial_tracking": 2,
    "drift": 1,
    "wrong_object": 1,
    "no_detection": 1,
    "fallback": 1,
}
DEFAULT_ABLATION_VARIANT = "grounding_dino+sam2_periodic_regrounding_every_10"
VARIANT_PRESETS = {
    DEFAULT_ABLATION_VARIANT: {
        "interval_frames": 10,
        "matching": "iou",
        "min_match_iou": 0.1,
        "record_frames": True,
    },
    "grounding_dino+sam2_periodic_regrounding_every_10_iou_0_3": {
        "interval_frames": 10,
        "matching": "iou",
        "min_match_iou": 0.3,
        "record_frames": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official periodic re-grounding ablation or finalize its comparison table.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True, help="Path to the locked subset manifest.")
    parser.add_argument("--baseline_dir", required=True, help="Directory containing the official baseline outputs and baseline_table.csv.")
    parser.add_argument("--output_dir", required=True, help="Directory for ablation outputs.")
    parser.add_argument(
        "--ablation_variant",
        default=DEFAULT_ABLATION_VARIANT,
        choices=sorted(VARIANT_PRESETS),
        help="Named ablation variant preset. Defaults to the original every-10-frames run with IoU 0.1.",
    )
    parser.add_argument(
        "--min_match_iou",
        type=float,
        default=None,
        help="Optional override for the minimum box-vs-mask IoU required before reseeding.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional subset cap for quick validation.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frame override.")
    parser.add_argument("--grounding_ckpt", default=None, help="Optional GroundingDINO checkpoint override.")
    parser.add_argument("--sam2_ckpt", default=None, help="Optional SAM2 checkpoint override.")
    parser.add_argument("--device", default=None, help="Optional runtime device override.")
    parser.add_argument(
        "--compare_ablation_dir",
        default=None,
        help="Optional path to another ablation run root for aggregate comparison output during finalize.",
    )
    parser.add_argument(
        "--finalize_reviewed",
        action="store_true",
        help="Skip inference and build ablation_delta_table.csv from a filled ablation_review_table.csv.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("clip_id", "")).strip(), str(row.get("video_path", "")).strip()


def _load_existing_reviews(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    return {
        _row_key(row): {
            "review_label": str(row.get("review_label", "")).strip(),
            "review_note": str(row.get("review_note", "")).strip(),
        }
        for row in _read_csv_rows(path)
    }


def _flatten_ablation_clip(clip: dict[str, Any], existing_reviews: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
    artifacts = clip.get("artifacts", {})
    video_overlay = str(artifacts.get("video_overlay", "")).strip()
    output_dir = str(Path(video_overlay).parent) if video_overlay else ""
    row = {
        "clip_id": str(clip.get("clip_id", "")).strip(),
        "video_path": str(clip.get("input_path", "")).strip(),
        "primary_tag": str(clip.get("primary_tag", "")).strip(),
        "prompt": str(clip.get("prompt", "")).strip(),
        "video_mode": str(artifacts.get("video_mode", "")).strip(),
        "runtime_sec": str(clip.get("runtime_sec", "")),
        "num_reground_attempts": str(clip.get("num_reground_attempts", "")),
        "num_reground_successes": str(clip.get("num_reground_successes", "")),
        "review_label": "",
        "review_note": "",
        "output_dir": output_dir,
    }
    review = existing_reviews.get(_row_key(row), {})
    row["review_label"] = review.get("review_label", "")
    row["review_note"] = review.get("review_note", "")
    return row


def _validate_review_rows(rows: list[dict[str, str]], table_name: str) -> None:
    missing_labels = [row["clip_id"] for row in rows if not row.get("review_label", "").strip()]
    if missing_labels:
        raise ValueError(
            f"{table_name} is missing review_label for: "
            + ", ".join(missing_labels[:5])
            + (" ..." if len(missing_labels) > 5 else "")
        )
    invalid = sorted({row["review_label"] for row in rows if row["review_label"] not in REVIEW_LABELS})
    if invalid:
        raise ValueError(f"Invalid review_label values in {table_name}: {invalid}. Allowed: {sorted(REVIEW_LABELS)}")
    missing_notes = [row["clip_id"] for row in rows if not row.get("review_note", "").strip()]
    if missing_notes:
        raise ValueError(
            f"{table_name} is missing review_note for: "
            + ", ".join(missing_notes[:5])
            + (" ..." if len(missing_notes) > 5 else "")
        )


def _count_review_labels(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("review_label", "")).strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _delta_label(baseline_label: str, reground_label: str) -> str:
    baseline_rank = LABEL_RANK[baseline_label]
    reground_rank = LABEL_RANK[reground_label]
    if reground_rank > baseline_rank:
        return "improved"
    if reground_rank < baseline_rank:
        return "worse"
    if reground_label == baseline_label:
        return "same"
    return "same"


def _resolve_variant_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(VARIANT_PRESETS[args.ablation_variant])
    if args.min_match_iou is not None:
        settings["min_match_iou"] = args.min_match_iou
    settings["ablation_variant"] = args.ablation_variant
    return settings


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.median(values)), 3)


def _reground_stats(clips: list[dict[str, Any]]) -> dict[str, float]:
    attempts = [float(clip.get("num_reground_attempts", 0)) for clip in clips]
    successes = [float(clip.get("num_reground_successes", 0)) for clip in clips]
    return {
        "mean_reground_attempts_per_clip": _mean(attempts),
        "median_reground_attempts_per_clip": _median(attempts),
        "mean_reground_successes_per_clip": _mean(successes),
        "median_reground_successes_per_clip": _median(successes),
    }


def _variant_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "ablation_variant": summary.get("ablation_variant", ""),
        "min_match_iou": summary.get("min_match_iou", 0.0),
        "num_selected": summary.get("num_selected", 0),
        "num_completed": summary.get("num_completed", 0),
        "review_counts": summary.get("review_counts", {}),
        "delta_counts": summary.get("delta_counts", {}),
        "mean_reground_successes_per_clip": summary.get("mean_reground_successes_per_clip", 0.0),
        "median_reground_successes_per_clip": summary.get("median_reground_successes_per_clip", 0.0),
    }


def _run_ablation(args: argparse.Namespace) -> int:
    from src.eval.eval_rvos import run_eval_subset
    from src.utils.io import load_project_config
    from src.utils.logger import configure_logger

    logger = configure_logger()
    config = load_project_config(
        ROOT / args.config,
        ROOT / "configs" / "grounding_dino.yaml",
        ROOT / "configs" / "sam2.yaml",
    )
    if args.max_frames is not None:
        config["runtime"]["max_frames"] = args.max_frames
    if args.grounding_ckpt is not None:
        config["grounding_dino"]["checkpoint_path"] = args.grounding_ckpt
    if args.sam2_ckpt is not None:
        config["sam2"]["checkpoint_path"] = args.sam2_ckpt
    if args.device is not None:
        config["runtime"]["device"] = args.device
        config["grounding_dino"]["device"] = args.device
        config["sam2"]["device"] = args.device

    variant_settings = _resolve_variant_settings(args)
    config.setdefault("ablation", {})
    config["ablation"]["variant"] = variant_settings["ablation_variant"]
    config["ablation"].setdefault("regrounding", {})
    config["ablation"]["regrounding"].update(
        {
            "enabled": True,
            "interval_frames": variant_settings["interval_frames"],
            "matching": variant_settings["matching"],
            "min_match_iou": variant_settings["min_match_iou"],
            "record_frames": variant_settings["record_frames"],
        }
    )

    output_root = Path(args.output_dir)
    summary = run_eval_subset(
        manifest_path=args.manifest,
        config=config,
        output_dir=output_root,
        limit=args.limit,
        summary_filename="ablation_summary.json",
        summary_overrides={
            "ablation_variant": config["ablation"]["variant"],
            "baseline_dir": str(Path(args.baseline_dir).resolve()),
            "min_match_iou": float(config["ablation"]["regrounding"]["min_match_iou"]),
        },
    )
    summary.update(_reground_stats(summary.get("clips", [])))
    _write_json(output_root / "ablation_summary.json", summary)

    review_table_path = output_root / "ablation_review_table.csv"
    existing_reviews = _load_existing_reviews(review_table_path)
    rows = [_flatten_ablation_clip(clip, existing_reviews) for clip in summary.get("clips", [])]
    _write_csv(
        review_table_path,
        rows,
        [
            "clip_id",
            "video_path",
            "primary_tag",
            "prompt",
            "video_mode",
            "runtime_sec",
            "num_reground_attempts",
            "num_reground_successes",
            "review_label",
            "review_note",
            "output_dir",
        ],
    )
    logger.info("Ablation run complete: %s", summary)
    logger.info("Review table template written to %s", review_table_path)
    return 0


def _finalize_reviewed(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir).resolve()
    baseline_root = Path(args.baseline_dir).resolve()
    ablation_summary_path = output_root / "ablation_summary.json"
    baseline_table_path = baseline_root / "baseline_table.csv"
    ablation_table_path = output_root / "ablation_review_table.csv"
    delta_table_path = output_root / "ablation_delta_table.csv"
    comparison_path = output_root / "ablation_variant_comparison.json"

    if not ablation_summary_path.exists():
        raise FileNotFoundError(f"Missing ablation summary: {ablation_summary_path}")
    if not baseline_table_path.exists():
        raise FileNotFoundError(f"Missing baseline review table: {baseline_table_path}")
    if not ablation_table_path.exists():
        raise FileNotFoundError(f"Missing ablation review table: {ablation_table_path}")

    baseline_rows = _read_csv_rows(baseline_table_path)
    ablation_rows = _read_csv_rows(ablation_table_path)
    _validate_review_rows(baseline_rows, "baseline_table.csv")
    _validate_review_rows(ablation_rows, "ablation_review_table.csv")

    baseline_by_key = {_row_key(row): row for row in baseline_rows}
    delta_rows: list[dict[str, Any]] = []
    delta_counts: dict[str, int] = {}
    for ablation_row in ablation_rows:
        key = _row_key(ablation_row)
        if key not in baseline_by_key:
            raise ValueError(f"Ablation row {ablation_row['clip_id']} is missing from baseline_table.csv")
        baseline_row = baseline_by_key[key]
        delta = _delta_label(
            str(baseline_row["review_label"]).strip(),
            str(ablation_row["review_label"]).strip(),
        )
        delta_counts[delta] = delta_counts.get(delta, 0) + 1
        delta_rows.append(
            {
                "clip_id": ablation_row["clip_id"],
                "primary_tag": ablation_row["primary_tag"],
                "prompt": ablation_row["prompt"],
                "baseline_review_label": baseline_row["review_label"],
                "reground_review_label": ablation_row["review_label"],
                "delta": delta,
                "baseline_runtime_sec": baseline_row["runtime_sec"],
                "reground_runtime_sec": ablation_row["runtime_sec"],
                "num_reground_attempts": ablation_row["num_reground_attempts"],
                "num_reground_successes": ablation_row["num_reground_successes"],
            }
        )

    _write_csv(
        delta_table_path,
        delta_rows,
        [
            "clip_id",
            "primary_tag",
            "prompt",
            "baseline_review_label",
            "reground_review_label",
            "delta",
            "baseline_runtime_sec",
            "reground_runtime_sec",
            "num_reground_attempts",
            "num_reground_successes",
        ],
    )

    summary = _load_json(ablation_summary_path)
    summary["review_counts"] = _count_review_labels(ablation_rows)
    summary["baseline_review_counts"] = _count_review_labels(baseline_rows)
    summary["delta_counts"] = delta_counts
    summary["ablation_review_table_path"] = str(ablation_table_path)
    summary["ablation_delta_table_path"] = str(delta_table_path)
    summary.update(_reground_stats(summary.get("clips", [])))
    _write_json(ablation_summary_path, summary)

    if args.compare_ablation_dir:
        compare_root = Path(args.compare_ablation_dir).resolve()
        compare_summary_path = compare_root / "ablation_summary.json"
        if not compare_summary_path.exists():
            raise FileNotFoundError(f"Missing comparison ablation summary: {compare_summary_path}")
        compare_summary = _load_json(compare_summary_path)
        comparison_payload = {
            "baseline_dir": str(baseline_root),
            "current_ablation_dir": str(output_root),
            "comparison_ablation_dir": str(compare_root),
            "current_variant": _variant_snapshot(summary),
            "comparison_variant": _variant_snapshot(compare_summary),
        }
        _write_json(comparison_path, comparison_payload)

    print(f"Wrote reviewed ablation summary to {ablation_summary_path}")
    print(f"Wrote ablation delta table to {delta_table_path}")
    if args.compare_ablation_dir:
        print(f"Wrote ablation comparison summary to {comparison_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.finalize_reviewed:
        return _finalize_reviewed(args)
    return _run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
