#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一入口：执行台风影响关键区域分析流程。"""

from __future__ import annotations

import argparse
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entry for typhoon impact physics analysis",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "compare",
        help="Run model receptive-field / physics comparison",
    )

    parser.set_defaults(command="compare")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "compare":
        from physics.comparison import run_physics_comparison
        run_physics_comparison()
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
