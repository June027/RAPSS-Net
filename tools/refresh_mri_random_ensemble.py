from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ensemble_predictions import (  # noqa: E402
    align_prediction_runs,
    build_summary_payload,
    evaluate_combinations,
    load_prediction_run,
    write_predictions_csv,
    write_ranking_csv,
)


def _unique_name(path: Path, used_names: set[str]) -> str:
    parts = path.parts
    if len(parts) >= 4:
        name = parts[-4]
    else:
        name = path.parent.name
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    safe = safe.strip("_") or "run"
    candidate = safe
    idx = 2
    while candidate in used_names:
        candidate = f"{safe}_{idx}"
        idx += 1
    used_names.add(candidate)
    return candidate


def collect_prediction_runs(root: Path, expected_rows: int | None):
    runs = []
    used_names: set[str] = set()
    for path in sorted(root.rglob("best_epoch_val_predictions.csv")):
        try:
            run = load_prediction_run(str(path), name=_unique_name(path, used_names))
        except Exception:
            continue
        if expected_rows is not None and len(run.ids) != int(expected_rows):
            continue
        runs.append(run)
    return runs


def _signature(run):
    return tuple(sorted(zip(run.ids, run.labels.tolist())))


def select_largest_aligned_group(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[_signature(run)].append(run)
    if not groups:
        return []
    return max(groups.values(), key=lambda group: (len(group), len(group[0].ids)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refresh soft-voting ensembles for MRI random-split validation predictions."
    )
    parser.add_argument(
        "--root",
        default="outputs/random",
        help="Experiment root to scan.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/random/ensemble_current",
        help="Output directory for ranking, summary, and best predictions.",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=183,
        help="Only include prediction files with this row count; use 0 to disable.",
    )
    parser.add_argument("--min_size", type=int, default=1)
    parser.add_argument("--max_size", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    expected_rows = None if int(args.expected_rows) <= 0 else int(args.expected_rows)
    runs = collect_prediction_runs(Path(args.root), expected_rows=expected_rows)
    aligned_runs = select_largest_aligned_group(runs)
    if len(aligned_runs) < 2:
        raise ValueError(
            f"Need at least two aligned prediction runs; found {len(aligned_runs)}."
        )

    ids, labels, aligned_probs = align_prediction_runs(aligned_runs)
    rows = evaluate_combinations(
        ids=ids,
        y_true=labels,
        aligned_probs=aligned_probs,
        min_size=int(args.min_size),
        max_size=int(args.max_size),
    )
    best_row = rows[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ranking_csv(str(output_dir / "ensemble_ranking.csv"), rows)
    write_predictions_csv(
        str(output_dir / "best_ensemble_predictions.csv"),
        ids=best_row["ids"],
        labels=best_row["labels"],
        probs=best_row["probs"],
    )
    summary = build_summary_payload(
        aligned_runs,
        rows,
        best_row,
        fixed_threshold=None,
    )
    summary["scanned_prediction_count"] = int(len(runs))
    summary["aligned_prediction_count"] = int(len(aligned_runs))
    with (output_dir / "ensemble_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    best = summary["best_combo"]
    print(
        "Best MRI ensemble | "
        f"AUC={best['auc']:.4f}, PR-AUC={best['pr_auc']:.4f}, "
        f"F1={best['f1']:.4f}, ACC={best['acc']:.4f}, "
        f"combo={best['combo_key']}"
    )
    print(f"Saved ensemble outputs to: {output_dir}")


if __name__ == "__main__":
    main()
