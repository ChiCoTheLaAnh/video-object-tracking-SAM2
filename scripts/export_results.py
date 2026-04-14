#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

SUCCESS_LABEL = "good_tracking"
REVIEW_LABELS = {
    SUCCESS_LABEL,
    "partial_tracking",
    "drift",
    "wrong_object",
    "no_detection",
    "fallback",
}
FAILURE_LABELS = {"drift", "wrong_object", "no_detection", "fallback"}
BORDERLINE_FAILURE_LABEL = "partial_tracking"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_run_summary(input_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [
        input_dir / "subset_run_summary.json",
        input_dir / "baseline_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, _load_json(candidate)
    raise FileNotFoundError(f"Could not find subset or baseline summary under {input_dir}")


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("clip_id", "")).strip(), str(row.get("video_path", "")).strip()


def _load_existing_reviews(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            _row_key(row): {
                "review_label": str(row.get("review_label", "")).strip(),
                "review_note": str(row.get("review_note", "")).strip(),
            }
            for row in reader
        }


def _flatten_clip(clip: dict[str, Any], existing_reviews: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
    artifacts = clip.get("artifacts", {})
    video_overlay = str(artifacts.get("video_overlay", "")).strip()
    output_dir = str(Path(video_overlay).parent) if video_overlay else ""
    base_row = {
        "clip_id": str(clip.get("clip_id", "")).strip(),
        "video_path": str(clip.get("input_path", "")).strip(),
        "primary_tag": str(clip.get("primary_tag", "")).strip(),
        "prompt": str(clip.get("prompt", "")).strip(),
        "video_mode": str(artifacts.get("video_mode", "")).strip(),
        "runtime_sec": str(clip.get("runtime_sec", "")),
        "review_label": "",
        "review_note": "",
        "output_dir": output_dir,
    }
    review = existing_reviews.get(_row_key(base_row), {})
    base_row["review_label"] = review.get("review_label", "")
    base_row["review_note"] = review.get("review_note", "")
    return base_row


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_review_rows(rows: list[dict[str, str]]) -> None:
    missing = [row["clip_id"] for row in rows if not row.get("review_label", "").strip()]
    if missing:
        raise ValueError(
            "baseline_table.csv is missing review_label for: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    invalid = sorted({row["review_label"] for row in rows if row["review_label"] not in REVIEW_LABELS})
    if invalid:
        raise ValueError(f"Invalid review_label values: {invalid}. Allowed: {sorted(REVIEW_LABELS)}")
    missing_notes = [row["clip_id"] for row in rows if not row.get("review_note", "").strip()]
    if missing_notes:
        raise ValueError(
            "baseline_table.csv is missing review_note for: "
            + ", ".join(missing_notes[:5])
            + (" ..." if len(missing_notes) > 5 else "")
        )


def _select_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    success_rows = [row for row in rows if row["review_label"] == SUCCESS_LABEL]
    if len(success_rows) < 5:
        raise ValueError(f"Need at least 5 `{SUCCESS_LABEL}` rows, found {len(success_rows)}")

    failure_rows = [row for row in rows if row["review_label"] in FAILURE_LABELS]
    borderline_rows = [row for row in rows if row["review_label"] == BORDERLINE_FAILURE_LABEL]
    if len(failure_rows) < 5:
        failure_rows = failure_rows + borderline_rows[: 5 - len(failure_rows)]
    if len(failure_rows) < 5:
        raise ValueError(
            "Need at least 5 failure examples from drift/wrong_object/no_detection/fallback "
            "or partial_tracking as fallback."
        )

    chosen = []
    for row in success_rows[:5]:
        chosen.append({**row, "example_bucket": "success"})
    for row in failure_rows[:5]:
        bucket_reason = "failure"
        if row["review_label"] == BORDERLINE_FAILURE_LABEL:
            bucket_reason = "failure_borderline"
        chosen.append({**row, "example_bucket": bucket_reason})
    return chosen


def _copy_artifact(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _export_frame(video_path: Path, frame_path: Path) -> str:
    if not video_path.exists():
        return ""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return ""
    total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    target_index = total_frames // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return ""
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), frame)
    return str(frame_path)


def _default_samples_dir(input_dir: Path) -> Path:
    if input_dir.parent.name == "quantitative" and input_dir.parent.parent.name == "results":
        return input_dir.parent.parent / "final_figures" / "baseline_samples"
    return input_dir / "baseline_samples"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export aggregate baseline outputs, a reviewed baseline table, and sample artifacts.")
    parser.add_argument("--input-dir", required=True, help="Subset or baseline run root containing subset_run_summary.json.")
    parser.add_argument("--samples-dir", default=None, help="Optional override for exported baseline sample artifacts.")
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="Require populated review labels/notes and export 5 success + 5 failure examples.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    summary_source_path, source_summary = _load_run_summary(input_dir)
    table_path = input_dir / "baseline_table.csv"
    examples_path = input_dir / "baseline_examples.csv"
    baseline_summary_path = input_dir / "baseline_summary.json"
    samples_dir = Path(args.samples_dir).resolve() if args.samples_dir else _default_samples_dir(input_dir)

    existing_reviews = _load_existing_reviews(table_path)
    clip_rows = [_flatten_clip(clip, existing_reviews) for clip in source_summary.get("clips", [])]
    fieldnames = [
        "clip_id",
        "video_path",
        "primary_tag",
        "prompt",
        "video_mode",
        "runtime_sec",
        "review_label",
        "review_note",
        "output_dir",
    ]
    _write_csv(table_path, clip_rows, fieldnames)

    review_counts: dict[str, int] = {}
    reviewed_rows = [row for row in clip_rows if row["review_label"]]
    for row in reviewed_rows:
        review_counts[row["review_label"]] = review_counts.get(row["review_label"], 0) + 1

    baseline_summary = dict(source_summary)
    baseline_summary.update(
        {
            "baseline_name": input_dir.name,
            "baseline_variant": "grounding_dino+sam2_no_regrounding",
            "source_summary_path": str(summary_source_path),
            "baseline_table_path": str(table_path),
            "review_counts": review_counts,
        }
    )
    _write_json(baseline_summary_path, baseline_summary)

    if not args.require_reviewed:
        print(f"Wrote template baseline table to {table_path}")
        print(f"Wrote aggregate baseline summary to {baseline_summary_path}")
        return 0

    _validate_review_rows(clip_rows)
    examples = _select_examples(clip_rows)

    exported_examples: list[dict[str, str]] = []
    for index, row in enumerate(examples, start=1):
        example_dir = samples_dir / ("success" if row["example_bucket"] == "success" else "failure")
        clip_output_dir = Path(row["output_dir"])
        overlay_src = clip_output_dir / "smoke_video_overlay.mp4"
        prefix = f"{index:02d}_{row['clip_id']}"
        overlay_dst = example_dir / f"{prefix}.mp4"
        frame_dst = example_dir / f"{prefix}.png"

        sample_video = _copy_artifact(overlay_src, overlay_dst)
        sample_frame = _export_frame(Path(sample_video), frame_dst) if sample_video else ""
        exported_examples.append(
            {
                **row,
                "sample_bucket": row["example_bucket"],
                "sample_video": sample_video,
                "sample_frame": sample_frame,
            }
        )

    _write_csv(
        examples_path,
        [
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "sample_bucket",
                    "clip_id",
                    "video_path",
                    "primary_tag",
                    "prompt",
                    "video_mode",
                    "runtime_sec",
                    "review_label",
                    "review_note",
                    "output_dir",
                    "sample_video",
                    "sample_frame",
                }
            }
            for row in exported_examples
        ],
        [
            "sample_bucket",
            "clip_id",
            "video_path",
            "primary_tag",
            "prompt",
            "video_mode",
            "runtime_sec",
            "review_label",
            "review_note",
            "output_dir",
            "sample_video",
            "sample_frame",
        ],
    )

    baseline_summary["baseline_examples_path"] = str(examples_path)
    baseline_summary["samples_dir"] = str(samples_dir)
    baseline_summary["num_examples"] = len(exported_examples)
    _write_json(baseline_summary_path, baseline_summary)

    print(f"Wrote aggregate baseline summary to {baseline_summary_path}")
    print(f"Wrote reviewed baseline table to {table_path}")
    print(f"Wrote example manifest to {examples_path}")
    print(f"Exported sample artifacts to {samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
