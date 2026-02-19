#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一入口：执行台风影响关键区域分析流程。"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from run_variable_importance_ig_ranking import run_gridpoint_importance_ranking


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entry for typhoon impact physics analysis",
    )
    subparsers = parser.add_subparsers(dest="command")

    ranking_parser = subparsers.add_parser(
        "ranking",
        help="Run IG + patch perturbation ranking pipeline",
    )
    ranking_parser.add_argument(
        "--json",
        action="store_true",
        help="Print compact JSON summary after the run",
    )

    parser.set_defaults(command="ranking", json=False)
    return parser


def _emit_summary(result: Dict[str, Any]) -> None:
    summary = {
        "target_vars": result.get("target_vars"),
        "top_k": result.get("top_k"),
        "top_n": result.get("top_n"),
        "patch_radius": result.get("patch_radius"),
        "output_csv": result.get("output_csv"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ranking":
        result = run_gridpoint_importance_ranking()
        if args.json:
            _emit_summary(result)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
