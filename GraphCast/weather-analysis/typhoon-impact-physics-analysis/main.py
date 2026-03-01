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
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    parser.parse_args(argv)
    from physics.comparison import run_physics_comparison_v2
    run_physics_comparison_v2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
