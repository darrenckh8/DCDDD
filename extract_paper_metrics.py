#!/usr/bin/env python3
"""Extract paper-ready training progress and results from training logs.

Usage:
  python extract_paper_metrics.py
  python extract_paper_metrics.py --outdir paper_results
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_LOGS = {
    "phase1_fold12": ROOT / "phase1-fold1-2.txt",
    "phase1_fold3": ROOT / "phase1-fold3.txt",
    "phase2": ROOT / "phase2.txt",
}


def _norm(text: str) -> str:
    """Light cleanup for mojibake-prone symbols in console logs."""
    return (
        text.replace("\ufeff", "")
        .replace("Â", "")
        .replace("â†’", "->")
        .replace("Î”", "Δ")
        .replace("â€”", "-")
        .replace("—", "-")
        .replace("â€“", "-")
    )


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [_norm(line.rstrip("\n")) for line in f]


def _metric_pairs(line: str) -> dict[str, float]:
    pairs = re.findall(
        r"([A-Za-z_]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        line,
    )
    return {k: float(v) for k, v in pairs}


def _parse_cm(lines: list[str], start_idx: int) -> tuple[int, int, int, int] | None:
    """Parse confusion matrix from two lines after a 'CM:' line."""
    if start_idx + 2 >= len(lines):
        return None
    row1 = [int(x) for x in re.findall(r"\d+", lines[start_idx + 1])]
    row2 = [int(x) for x in re.findall(r"\d+", lines[start_idx + 2])]
    if len(row1) >= 2 and len(row2) >= 2:
        tn, fp = row1[0], row1[1]
        fn, tp = row2[0], row2[1]
        return tn, fp, fn, tp
    return None


def parse_training_progress(log_path: Path, phase: str) -> list[dict[str, Any]]:
    lines = _read_lines(log_path)
    rows: list[dict[str, Any]] = []
    current_fold: int | None = None
    current_epoch: int | None = None
    current_total_epochs: int | None = None

    for line in lines:
        m_fold = re.search(r"\bFOLD\s+(\d+)\s*/\s*(\d+)\b", line, flags=re.IGNORECASE)
        if m_fold:
            current_fold = int(m_fold.group(1))
            continue

        m_epoch = re.search(r"\bEpoch\s+(\d+)\s*/\s*(\d+)\b", line, flags=re.IGNORECASE)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))
            current_total_epochs = int(m_epoch.group(2))
            continue

        # Keras prints one short line + one full val_* line for each epoch.
        # We use the val_* line to keep one record per epoch.
        if current_epoch is None or "val_auc:" not in line:
            continue

        metrics = _metric_pairs(line)
        if "val_auc" not in metrics:
            continue

        m_speed = re.search(
            r"(\d+)\s*/\s*\d+.*?(\d+)s\s+(\d+)ms/step",
            line,
            flags=re.IGNORECASE,
        )
        steps = int(m_speed.group(1)) if m_speed else None
        epoch_seconds = int(m_speed.group(2)) if m_speed else None
        ms_per_step = int(m_speed.group(3)) if m_speed else None

        rows.append(
            {
                "source_log": log_path.name,
                "phase": phase,
                "fold": current_fold,
                "epoch": current_epoch,
                "total_epochs": current_total_epochs,
                "steps": steps,
                "epoch_seconds": epoch_seconds,
                "ms_per_step": ms_per_step,
                "auc": metrics.get("auc"),
                "hard_accuracy": metrics.get("hard_accuracy"),
                "loss": metrics.get("loss"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "val_auc": metrics.get("val_auc"),
                "val_hard_accuracy": metrics.get("val_hard_accuracy"),
                "val_loss": metrics.get("val_loss"),
                "val_precision": metrics.get("val_precision"),
                "val_recall": metrics.get("val_recall"),
                "learning_rate": metrics.get("learning_rate"),
            }
        )
        current_epoch = None

    return rows


def parse_fold_result_blocks(log_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for log_path in log_paths:
        lines = _read_lines(log_path)
        for i, line in enumerate(lines):
            m = re.match(r"\s*Fold\s+(\d+)\s+@thr=([0-9.]+)", line)
            if not m:
                continue

            fold = int(m.group(1))
            thr = float(m.group(2))
            key = (fold, f"{thr:.3f}")
            if key in seen:
                continue
            seen.add(key)

            row: dict[str, Any] = {
                "source_log": log_path.name,
                "fold": fold,
                "threshold": thr,
                "auc": None,
                "prauc": None,
                "accuracy": None,
                "accuracy_05": None,
                "f1": None,
                "f1_05": None,
                "precision": None,
                "recall": None,
                "recall_05": None,
                "cm_tn": None,
                "cm_fp": None,
                "cm_fn": None,
                "cm_tp": None,
            }

            for j in range(i + 1, min(i + 18, len(lines))):
                s = lines[j].strip()
                if s.startswith("AUC:"):
                    vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                    if len(vals) >= 2:
                        row["auc"], row["prauc"] = vals[0], vals[1]
                elif s.startswith("Acc:"):
                    vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                    if len(vals) >= 2:
                        row["accuracy"], row["accuracy_05"] = vals[0], vals[1]
                elif s.startswith("F1:"):
                    vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                    if len(vals) >= 2:
                        row["f1"], row["f1_05"] = vals[0], vals[1]
                elif s.startswith("Prec:"):
                    vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                    if vals:
                        row["precision"] = vals[0]
                elif s.startswith("Rec:"):
                    vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                    if len(vals) >= 2:
                        row["recall"], row["recall_05"] = vals[0], vals[1]
                elif s.startswith("CM:"):
                    cm = _parse_cm(lines, j)
                    if cm is not None:
                        row["cm_tn"], row["cm_fp"], row["cm_fn"], row["cm_tp"] = cm
            rows.append(row)
    return sorted(rows, key=lambda r: int(r["fold"]))


def parse_cv_summary(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    lines = _read_lines(log_path)
    summary_rows: list[dict[str, Any]] = []
    overview: dict[str, Any] = {}
    fold_table_rows: list[dict[str, Any]] = []

    metric_re = re.compile(
        r"^\s*(accuracy|f1|auc|prauc|precision|recall):\s*([0-9.]+)\s*[^0-9A-Za-z]+\s*([0-9.]+)\s*\(min=([0-9.]+)\s+max=([0-9.]+)\)",
        re.IGNORECASE,
    )

    in_fold_table = False
    for i, line in enumerate(lines):
        m_metric = metric_re.match(line)
        if m_metric:
            summary_rows.append(
                {
                    "metric": m_metric.group(1).lower(),
                    "mean": float(m_metric.group(2)),
                    "std": float(m_metric.group(3)),
                    "min": float(m_metric.group(4)),
                    "max": float(m_metric.group(5)),
                }
            )
            continue

        if "Thresholds:" in line:
            vals = [float(x) for x in re.findall(r"\d+\.\d+", line)]
            overview["thresholds"] = ",".join(f"{x:.3f}" for x in vals)
            continue

        if "Mean threshold:" in line:
            vals = [float(x) for x in re.findall(r"\d+\.\d+", line)]
            if vals:
                overview["mean_threshold"] = vals[0]
            continue

        if re.match(r"^\s*(accuracy|f1|recall):", line) and "->" in line:
            metric = re.match(r"^\s*(accuracy|f1|recall):", line).group(1).lower()
            nums = [float(x) for x in re.findall(r"[+-]?\d+\.\d+", line)]
            if len(nums) >= 3:
                overview[f"{metric}_baseline_05"] = nums[0]
                overview[f"{metric}_optimized_thr"] = nums[1]
                overview[f"{metric}_delta"] = nums[2]
            continue

        if line.strip().startswith("Aggregate CM:"):
            cm = _parse_cm(lines, i)
            if cm is not None:
                overview["aggregate_cm_tn"], overview["aggregate_cm_fp"], overview["aggregate_cm_fn"], overview["aggregate_cm_tp"] = cm
            continue

        if line.strip().startswith("Fold |"):
            in_fold_table = True
            continue

        if in_fold_table:
            if not line.strip():
                in_fold_table = False
                continue
            if set(line.strip()) == {"-"}:
                continue
            m_row = re.match(
                r"^\s*(\d+)\s+\|\s*([0-9.]+)\s+\|\s*([0-9.]+)\s+\|\s*([0-9.]+)\s+\|\s*([0-9.]+)\s+\|\s*([0-9.]+)\s+\|\s*([0-9.]+)\s+\|\s*([0-9.]+)\s+\|\s*(.+)$",
                line,
            )
            if m_row:
                fold_table_rows.append(
                    {
                        "fold": int(m_row.group(1)),
                        "threshold": float(m_row.group(2)),
                        "accuracy": float(m_row.group(3)),
                        "f1": float(m_row.group(4)),
                        "auc": float(m_row.group(5)),
                        "prauc": float(m_row.group(6)),
                        "precision": float(m_row.group(7)),
                        "recall": float(m_row.group(8)),
                        "subjects_preview": m_row.group(9).strip(),
                    }
                )
            continue

        m_best = re.search(r"Best fold model \(fold\s+(\d+),\s*AUC=([0-9.]+)\):\s*(.+)$", line)
        if m_best:
            overview["best_fold"] = int(m_best.group(1))
            overview["best_fold_auc"] = float(m_best.group(2))
            overview["best_fold_model_path"] = m_best.group(3).strip()

    return summary_rows, overview, fold_table_rows


def parse_phase2_results(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    lines = _read_lines(log_path)
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    perf_rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def _new_row(split: str, label: str, thr: float) -> dict[str, Any]:
        return {
            "split": split,
            "label": label,
            "threshold": thr,
            "auc": None,
            "prauc": None,
            "accuracy": None,
            "accuracy_05": None,
            "f1": None,
            "f1_05": None,
            "precision": None,
            "recall": None,
            "recall_05": None,
            "cm_tn": None,
            "cm_fp": None,
            "cm_fn": None,
            "cm_tp": None,
            "support": None,
        }

    for i, line in enumerate(lines):
        if "Final model splits:" in line:
            m = re.search(
                r"Final model splits:\s*(\d+)\s*train\s*\|\s*(\d+)\s*val\s*\|\s*(\d+)\s*thr-tune\s*\|\s*(\d+)\s*report",
                line,
                flags=re.IGNORECASE,
            )
            if m:
                metadata["train_subjects"] = int(m.group(1))
                metadata["val_subjects"] = int(m.group(2))
                metadata["thr_tune_subjects"] = int(m.group(3))
                metadata["report_subjects"] = int(m.group(4))
            continue

        if "Threshold holdout:" in line and "sequences" in line:
            nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", line)]
            if nums:
                metadata["threshold_holdout_sequences"] = nums[0]
            continue

        if "Deployment threshold:" in line:
            vals = [float(x) for x in re.findall(r"\d+\.\d+", line)]
            if vals:
                metadata["deployment_threshold"] = vals[0]
            continue

        m_hdr = re.match(
            r"^\s*(Held-out report set.*?|Validation set.*?|Threshold holdout.*?)\s+@thr=([0-9.]+)",
            line,
        )
        if m_hdr:
            label = m_hdr.group(1).strip()
            thr = float(m_hdr.group(2))
            label_lower = label.lower()
            if label_lower.startswith("held-out report"):
                split = "report_honest"
            elif label_lower.startswith("validation"):
                split = "validation_tainted"
            else:
                split = "threshold_holdout_tainted"
            current = _new_row(split=split, label=label, thr=thr)
            rows.append(current)
            continue

        if current is not None:
            s = line.strip()
            if s.startswith("AUC:"):
                vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                if len(vals) >= 2:
                    current["auc"], current["prauc"] = vals[0], vals[1]
                continue
            if s.startswith("Acc:"):
                vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                if len(vals) >= 2:
                    current["accuracy"], current["accuracy_05"] = vals[0], vals[1]
                continue
            if s.startswith("F1:"):
                vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                if len(vals) >= 2:
                    current["f1"], current["f1_05"] = vals[0], vals[1]
                continue
            if s.startswith("Prec:"):
                vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                if vals:
                    current["precision"] = vals[0]
                continue
            if s.startswith("Rec:"):
                vals = [float(x) for x in re.findall(r"\d+\.\d+", s)]
                if len(vals) >= 2:
                    current["recall"], current["recall_05"] = vals[0], vals[1]
                continue
            if s.startswith("CM:"):
                cm = _parse_cm(lines, i)
                if cm is not None:
                    current["cm_tn"], current["cm_fp"], current["cm_fn"], current["cm_tp"] = cm
                    current["support"] = sum(cm)
                continue
            if s.startswith("weighted avg"):
                current = None
                continue

        # Deployment artifacts and latency
        m_size = re.match(
            r"^(Float16 TFLite|INT8 TFLite.*?):\s*(.+)\s+\(([0-9.]+)\s*KB\)$",
            line.strip(),
        )
        if m_size:
            perf_rows.append(
                {
                    "category": "artifact_size",
                    "name": m_size.group(1),
                    "path_or_variant": m_size.group(2),
                    "avg_ms": None,
                    "p50_ms": None,
                    "p95_ms": None,
                    "size_kb": float(m_size.group(3)),
                }
            )
            continue

        m_lat = re.match(
            r"^\s*(Float16|INT8-hybrid):\s*avg=([0-9.]+)ms\s+p50=([0-9.]+)ms\s+p95=([0-9.]+)ms",
            line,
            flags=re.IGNORECASE,
        )
        if m_lat:
            perf_rows.append(
                {
                    "category": "latency",
                    "name": m_lat.group(1),
                    "path_or_variant": None,
                    "avg_ms": float(m_lat.group(2)),
                    "p50_ms": float(m_lat.group(3)),
                    "p95_ms": float(m_lat.group(4)),
                    "size_kb": None,
                }
            )

    return rows, metadata, perf_rows


def best_epochs(training_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in training_rows:
        key = (row["phase"], row["fold"])
        by_group.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (phase, fold), rows in sorted(by_group.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        rows_sorted = sorted(rows, key=lambda r: int(r["epoch"]))
        best = max(rows_sorted, key=lambda r: float(r["val_auc"]))
        last = rows_sorted[-1]
        out.append(
            {
                "phase": phase,
                "fold": fold,
                "best_epoch_by_val_auc": best["epoch"],
                "best_val_auc": best["val_auc"],
                "best_val_accuracy": best["val_hard_accuracy"],
                "best_val_loss": best["val_loss"],
                "last_epoch": last["epoch"],
                "last_val_auc": last["val_auc"],
                "last_val_accuracy": last["val_hard_accuracy"],
                "epochs_ran": len(rows_sorted),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write header-only empty file if fieldnames are provided.
        if fieldnames:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_kv_csv(path: Path, data: dict[str, Any]) -> None:
    rows = [{"key": k, "value": data[k]} for k in sorted(data.keys())]
    write_csv(path, rows, fieldnames=["key", "value"])


def write_summary_md(
    out_path: Path,
    cv_fold_rows: list[dict[str, Any]],
    cv_summary_rows: list[dict[str, Any]],
    phase2_rows: list[dict[str, Any]],
    best_epoch_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Paper-Ready Extracted Results")
    lines.append("")

    lines.append("## Cross-Validation Fold Results")
    lines.append("")
    lines.append("| Fold | Thr | Acc | F1 | AUC | PR-AUC | Prec | Rec |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(cv_fold_rows, key=lambda x: int(x["fold"])):
        lines.append(
            f"| {r['fold']} | {r['threshold']:.3f} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
            f"{r['auc']:.4f} | {r['prauc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )
    lines.append("")

    lines.append("## CV Mean +/- Std")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in sorted(cv_summary_rows, key=lambda x: x["metric"]):
        lines.append(
            f"| {r['metric']} | {r['mean']:.4f} | {r['std']:.4f} | {r['min']:.4f} | {r['max']:.4f} |"
        )
    lines.append("")

    lines.append("## Phase-2 Final Metrics")
    lines.append("")
    lines.append("| Split | Thr | Acc | F1 | AUC | PR-AUC | Prec | Rec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in phase2_rows:
        lines.append(
            f"| {r['split']} | {r['threshold']:.3f} | {r['accuracy']:.4f} | {r['f1']:.4f} | "
            f"{r['auc']:.4f} | {r['prauc']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )
    lines.append("")

    lines.append("## Best Epochs (By Val AUC)")
    lines.append("")
    lines.append("| Phase | Fold | Best Epoch | Best Val AUC | Last Epoch | Last Val AUC | Epochs Ran |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in best_epoch_rows:
        fold = "" if r["fold"] is None else str(r["fold"])
        lines.append(
            f"| {r['phase']} | {fold} | {r['best_epoch_by_val_auc']} | {r['best_val_auc']:.4f} | "
            f"{r['last_epoch']} | {r['last_val_auc']:.4f} | {r['epochs_ran']} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training progress and paper-ready metrics from logs.")
    parser.add_argument("--outdir", default="paper_results", help="Output directory (relative to repo root).")
    args = parser.parse_args()

    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    missing = [p for p in DEFAULT_LOGS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected log files: {', '.join(str(m) for m in missing)}")

    training_rows: list[dict[str, Any]] = []
    training_rows.extend(parse_training_progress(DEFAULT_LOGS["phase1_fold12"], phase="phase1_cv"))
    training_rows.extend(parse_training_progress(DEFAULT_LOGS["phase1_fold3"], phase="phase1_cv"))
    training_rows.extend(parse_training_progress(DEFAULT_LOGS["phase2"], phase="phase2_final"))
    training_rows.sort(key=lambda r: (r["phase"], -1 if r["fold"] is None else int(r["fold"]), int(r["epoch"])))

    fold_result_rows = parse_fold_result_blocks([DEFAULT_LOGS["phase1_fold12"], DEFAULT_LOGS["phase1_fold3"]])
    cv_summary_rows, cv_overview, cv_fold_table_rows = parse_cv_summary(DEFAULT_LOGS["phase1_fold3"])
    phase2_rows, phase2_metadata, perf_rows = parse_phase2_results(DEFAULT_LOGS["phase2"])
    best_epoch_rows = best_epochs(training_rows)

    # Prefer the explicit CV table from summary if present.
    cv_fold_rows = cv_fold_table_rows if cv_fold_table_rows else fold_result_rows

    write_csv(
        outdir / "training_progress.csv",
        training_rows,
        fieldnames=[
            "source_log",
            "phase",
            "fold",
            "epoch",
            "total_epochs",
            "steps",
            "epoch_seconds",
            "ms_per_step",
            "auc",
            "hard_accuracy",
            "loss",
            "precision",
            "recall",
            "val_auc",
            "val_hard_accuracy",
            "val_loss",
            "val_precision",
            "val_recall",
            "learning_rate",
        ],
    )

    write_csv(
        outdir / "cv_fold_results.csv",
        cv_fold_rows,
        fieldnames=[
            "fold",
            "threshold",
            "accuracy",
            "f1",
            "auc",
            "prauc",
            "precision",
            "recall",
            "subjects_preview",
        ]
        if cv_fold_rows and "subjects_preview" in cv_fold_rows[0]
        else [
            "source_log",
            "fold",
            "threshold",
            "auc",
            "prauc",
            "accuracy",
            "accuracy_05",
            "f1",
            "f1_05",
            "precision",
            "recall",
            "recall_05",
            "cm_tn",
            "cm_fp",
            "cm_fn",
            "cm_tp",
        ],
    )

    write_csv(
        outdir / "cv_summary.csv",
        cv_summary_rows,
        fieldnames=["metric", "mean", "std", "min", "max"],
    )
    write_kv_csv(outdir / "cv_overview.csv", cv_overview)

    write_csv(
        outdir / "phase2_results.csv",
        phase2_rows,
        fieldnames=[
            "split",
            "label",
            "threshold",
            "auc",
            "prauc",
            "accuracy",
            "accuracy_05",
            "f1",
            "f1_05",
            "precision",
            "recall",
            "recall_05",
            "cm_tn",
            "cm_fp",
            "cm_fn",
            "cm_tp",
            "support",
        ],
    )
    write_kv_csv(outdir / "phase2_metadata.csv", phase2_metadata)
    write_csv(
        outdir / "deployment_perf.csv",
        perf_rows,
        fieldnames=["category", "name", "path_or_variant", "size_kb", "avg_ms", "p50_ms", "p95_ms"],
    )

    write_csv(
        outdir / "best_epochs.csv",
        best_epoch_rows,
        fieldnames=[
            "phase",
            "fold",
            "best_epoch_by_val_auc",
            "best_val_auc",
            "best_val_accuracy",
            "best_val_loss",
            "last_epoch",
            "last_val_auc",
            "last_val_accuracy",
            "epochs_ran",
        ],
    )

    write_summary_md(
        out_path=outdir / "summary.md",
        cv_fold_rows=cv_fold_rows,
        cv_summary_rows=cv_summary_rows,
        phase2_rows=phase2_rows,
        best_epoch_rows=best_epoch_rows,
    )

    print(f"Wrote extracted outputs to: {outdir}")
    print(f"- training_progress.csv ({len(training_rows)} rows)")
    print(f"- cv_fold_results.csv ({len(cv_fold_rows)} rows)")
    print(f"- cv_summary.csv ({len(cv_summary_rows)} rows)")
    print(f"- phase2_results.csv ({len(phase2_rows)} rows)")
    print(f"- best_epochs.csv ({len(best_epoch_rows)} rows)")
    print("- summary.md")


if __name__ == "__main__":
    main()
