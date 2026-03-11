from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _expand_report_paths(inputs: Sequence[str | Path]) -> List[Path]:
    paths: List[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            paths.extend(sorted(path.rglob("dlmsf_track_patch_alignment.json")))
        elif path.is_file():
            paths.append(path)
    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    return unique_paths


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(metrics: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    mean = float(np.mean(arr[finite]))
    std = float(np.std(arr[finite]))
    out = np.zeros_like(arr)
    if std < 1e-12:
        return out
    out[finite] = (arr[finite] - mean) / std
    return out


def _matches_subset(report: Dict[str, Any], report_path: Path, subset: str | None) -> bool:
    if not subset:
        return True
    subset_lc = subset.lower().strip()
    candidates = [
        str(report.get("source_pipeline", "")),
        str(report.get("subset", "")),
        str(report.get("dataset_subset", "")),
        str(report_path),
    ]
    return any(subset_lc in candidate.lower() for candidate in candidates)


def collect_track_patch_rows(
    inputs: Sequence[str | Path],
    *,
    main_case: str = "along_p3",
    subset: str | None = "swe",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report_path in _expand_report_paths(inputs):
        report = _load_report(report_path)
        if not _matches_subset(report, report_path, subset):
            continue
        case = report.get("cases", {}).get(main_case)
        if not isinstance(case, dict):
            continue
        metrics = case.get("metrics", {})
        deletion = case.get("deletion", {})
        rows.append(
            {
                "report_path": str(report_path),
                "report_name": report_path.name,
                "source_pipeline": str(report.get("source_pipeline", "")),
                "main_case": main_case,
                "spearman_rho": _metric_value(metrics, "spearman_rho"),
                "iou_topk": _metric_value(metrics, "iou_topk", "topk_overlap"),
                "topk_k": int(metrics.get("topk_k", report.get("topk_k", 0))),
                "high_ig_aopc": float(deletion.get("high_ig_aopc", np.nan)),
                "low_ig_aopc": float(deletion.get("low_ig_aopc", np.nan)),
                "random_mean_aopc": float(deletion.get("random_mean_aopc", np.nan)),
            }
        )
    return rows


def rank_track_patch_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> Dict[str, Any]:
    if not rows:
        return {
            "metadata": {
                "count_total": 0,
                "count_hard_filtered": 0,
                "top_n": int(top_n),
            },
            "rows": [],
        }

    enriched_rows: List[Dict[str, Any]] = []
    for row in rows:
        hard_pass = (
            float(row["high_ig_aopc"]) > float(row["random_mean_aopc"])
            and float(row["high_ig_aopc"]) > float(row["low_ig_aopc"])
        )
        new_row = dict(row)
        new_row["deletion_advantage"] = float(row["high_ig_aopc"]) - float(row["random_mean_aopc"])
        new_row["hard_filter_pass"] = bool(hard_pass)
        enriched_rows.append(new_row)

    hard_rows = [row for row in enriched_rows if row["hard_filter_pass"]]

    spearman_z = _zscore(np.asarray([row["spearman_rho"] for row in hard_rows], dtype=np.float64))
    iou_z = _zscore(np.asarray([row["iou_topk"] for row in hard_rows], dtype=np.float64))
    advantage_z = _zscore(np.asarray([row["deletion_advantage"] for row in hard_rows], dtype=np.float64))
    ranked_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(hard_rows):
        new_row = dict(row)
        new_row["z_spearman"] = float(spearman_z[idx]) if idx < spearman_z.size else 0.0
        new_row["z_iou"] = float(iou_z[idx]) if idx < iou_z.size else 0.0
        new_row["z_deletion_advantage"] = float(advantage_z[idx]) if idx < advantage_z.size else 0.0
        new_row["score"] = (
            0.35 * new_row["z_spearman"]
            + 0.35 * new_row["z_iou"]
            + 0.15 * new_row["z_deletion_advantage"]
        )
        ranked_rows.append(new_row)

    ranked_rows.sort(key=lambda row: (row["score"], row["spearman_rho"], row["iou_topk"]), reverse=True)
    selected_rows = ranked_rows[: max(0, int(top_n))]
    for rank, row in enumerate(selected_rows, start=1):
        row["rank"] = rank

    return {
        "metadata": {
            "count_total": len(enriched_rows),
            "count_hard_filtered": len(hard_rows),
            "top_n": int(top_n),
        },
        "rows": selected_rows,
    }


def write_ranked_rows_csv(rows: Iterable[Dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames = [
        "rank",
        "report_name",
        "report_path",
        "score",
        "spearman_rho",
        "iou_topk",
        "high_ig_aopc",
        "random_mean_aopc",
        "low_ig_aopc",
        "deletion_advantage",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank DLMSF-vs-IG track-patch reports.")
    parser.add_argument("inputs", nargs="+", help="Report JSON files or directories to scan recursively.")
    parser.add_argument("--main-case", default="along_p3", help="Case key to evaluate. Default: along_p3")
    parser.add_argument("--subset", default="swe", help="Subset filter token. Default: swe")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = collect_track_patch_rows(
        args.inputs,
        main_case=args.main_case,
        subset=args.subset,
    )
    ranked = rank_track_patch_rows(
        rows,
        top_n=args.top_n,
    )

    payload_text = json.dumps(ranked, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload_text, encoding="utf-8")
    if args.output_csv is not None:
        write_ranked_rows_csv(ranked["rows"], args.output_csv)

    print(payload_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
