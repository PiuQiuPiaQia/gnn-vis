#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cyclone_points import CYCLONE_YAAS_CENTERS  # noqa: E402
from preprocess.wb2_graphcast_export import (  # noqa: E402
    GRAPHCAST_LOW_RES_STORE,
    SOLAR_STORE,
    build_explicit_window,
    build_track_window,
    default_output_path,
    export_tauktae_graphcast_low_res,
    infer_steps_from_window,
)


DEFAULT_YAAS_STEPS = len(CYCLONE_YAAS_CENTERS) - 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull either the default CYCLONE_YAAS_CENTERS window or an explicit "
            "UTC time range from WeatherBench2 and export it as a GraphCast-compatible "
            "low-resolution sample dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target NetCDF path. Defaults to the GraphCast low-res filename under /root/autodl-tmp/dataset.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_YAAS_STEPS,
        help="Number of 6-hour target steps in the output file name and relative time axis. Ignored when --start/--end are provided.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Explicit UTC start time. Accepted formats: 'YYYY-MM-DD HHZ' or 'MM/DD/YYYY HHZ'. Must be used together with --end.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Explicit UTC end time. Accepted formats: 'YYYY-MM-DD HHZ' or 'MM/DD/YYYY HHZ'. Must be used together with --start.",
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
        "--no-progress",
        action="store_true",
        help="Disable terminal progress output while loading WeatherBench2 data.",
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
    if (args.start is None) != (args.end is None):
        print("Export failed: --start and --end must be provided together.", file=sys.stderr)
        return 1

    if args.start is not None and args.end is not None:
        try:
            abs_times = build_explicit_window(args.start, args.end)
        except ValueError as exc:
            print(f"Export failed: {exc}", file=sys.stderr)
            return 1
        steps = infer_steps_from_window(abs_times)
    else:
        steps = args.steps
        abs_times = build_track_window(CYCLONE_YAAS_CENTERS, steps=steps)

    output_path = args.output if args.output is not None else default_output_path(steps=steps, abs_times=abs_times)

    print("CYCLONE_YAAS_CENTERS export window:", flush=True)
    for idx, timestamp in enumerate(abs_times):
        print(f"  [{idx}] {timestamp:%Y-%m-%d %H:%M} UTC", flush=True)
    print(f"Output: {output_path}", flush=True)

    if args.dry_run:
        return 0

    try:
        final_path = export_tauktae_graphcast_low_res(
            output_path=output_path,
            steps=steps,
            force=args.force,
            low_res_store=args.low_res_store,
            solar_store=args.solar_store,
            abs_times=abs_times,
            show_progress=not args.no_progress,
        )
    except (FileExistsError, ModuleNotFoundError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote GraphCast dataset: {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
