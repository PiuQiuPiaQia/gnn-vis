#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.wb2_graphcast_export import (  # noqa: E402
    CYCLONE_TAUKTAE_CENTERS,
    GRAPHCAST_LOW_RES_STORE,
    SOLAR_STORE,
    build_track_window,
    default_output_path,
    export_tauktae_graphcast_low_res,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull the CYCLONE_TAUKTAE_CENTERS WeatherBench2 window and export it "
            "as a GraphCast-compatible low-resolution sample dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target NetCDF path. Defaults to the GraphCast low-res filename under /root/data/dataset.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of 6-hour target steps in the output file name and relative time axis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved time window and output path without downloading data.",
    )
    parser.add_argument(
        "--low-res-store",
        default=GRAPHCAST_LOW_RES_STORE,
        help="WeatherBench2 1.0-degree ERA5 store used for all fields except solar forcing.",
    )
    parser.add_argument(
        "--solar-store",
        default=SOLAR_STORE,
        help="WeatherBench2 store used to source toa_incident_solar_radiation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output if args.output is not None else default_output_path(steps=args.steps)
    abs_times = build_track_window(CYCLONE_TAUKTAE_CENTERS, steps=args.steps)

    print("CYCLONE_TAUKTAE_CENTERS export window:", flush=True)
    for idx, timestamp in enumerate(abs_times):
        print(f"  [{idx}] {timestamp:%Y-%m-%d %H:%M} UTC", flush=True)
    print(f"Output: {output_path}", flush=True)

    if args.dry_run:
        return 0

    try:
        final_path = export_tauktae_graphcast_low_res(
            output_path=output_path,
            steps=args.steps,
            force=args.force,
            low_res_store=args.low_res_store,
            solar_store=args.solar_store,
        )
    except (FileExistsError, ModuleNotFoundError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote GraphCast dataset: {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
