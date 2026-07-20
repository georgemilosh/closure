#!/usr/bin/env python
"""Band-resolved spectral scalars from a single field component (default Ez).

Companion to ``closure-diagnostics bands``, which decomposes the full E (or B)
vector spectrum: there a run with a large coherent in-plane Ey dilutes
``grid_frac`` even when the Ez panel looks noisy by eye. This script computes
the same recon/wave/grid bands from one scalar component so the fractions
match the single-panel visual impression.

The output CSV mirrors ``bands_menura.csv`` exactly (the ``field`` column
holds the component name, e.g. ``Ez``), so ``closure-diagnostics overlay``
consumes it unchanged, e.g.::

    closure-diagnostics overlay diagnostics/<folder>/R5/bands_ez_menura.csv \
        --x time --y grid_frac --group-by run

Usage matches the ``bands`` subcommand, with ``--component`` replacing
``--field``::

    python scripts/bands_ez.py --backend menura --files-path <run_dir> \
        --component Ez --choose-times all --output-csv out.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from closure.diagnostics import export_bands_dataframe
from closure.diagnostics_cli import (
    _add_load_options,
    _configure_logging,
    _load_for_command,
    _resolve_experiments,
    _write_csv,
)

logger = logging.getLogger("bands_ez")


def _scalar_bands_one_experiment(args: argparse.Namespace, experiment: str) -> pd.DataFrame:
    """Load one experiment and return its scalar-component band frame.

    Module-level (picklable) so it can be dispatched to a process pool,
    mirroring ``diagnostics_cli._bands_one_experiment``.
    """
    logger.info(
        "Loading %s with backend=%s, choose_times=%s, component=%s",
        experiment,
        args.backend,
        args.choose_times,
        args.component,
    )
    data, X, Y, qom, times = _load_for_command(args, experiment)
    try:
        comp = np.asarray(data[args.component])
    except KeyError as exc:
        raise SystemExit(
            f"Component {args.component!r} not in loaded data for {experiment}; "
            f"available fields include: {sorted(k for k in data if isinstance(data[k], np.ndarray))}"
        ) from exc
    # A vector power spectrum is the sum of its component spectra, so routing
    # (comp, 0, 0) through the vector path yields the scalar spectrum exactly
    # while reusing the tested radial binning in export_bands_dataframe.
    zeros = np.zeros_like(comp)
    frame = export_bands_dataframe(
        {"Ex": comp, "Ey": zeros, "Ez": zeros},
        X,
        Y,
        times,
        run_name=experiment,
        field="E",
        f_lo=args.f_lo,
        f_hi=args.f_hi,
    )
    frame["field"] = args.component
    logger.info("Computed %s band diagnostics for %s: %d rows", args.component, experiment, len(frame))
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export band-resolved spectral scalars of a single field component to CSV",
    )
    _add_load_options(parser, default_choose_times="all")
    parser.add_argument("--output-csv", default="diagnostics/bands_ez.csv", help="Output CSV path")
    parser.add_argument(
        "--component",
        default="Ez",
        help="Scalar field component to analyze, any loaded 2D field key (default: Ez)",
    )
    parser.add_argument(
        "--f-lo",
        type=float,
        default=0.15,
        help="recon/wave band edge as a fraction of the Nyquist wavenumber (default: 0.15)",
    )
    parser.add_argument(
        "--f-hi",
        type=float,
        default=0.80,
        help="wave/grid band edge as a fraction of the Nyquist wavenumber (default: 0.80)",
    )
    parser.add_argument(
        "--csv-mode",
        choices=["append", "replace"],
        default="append",
        help="Whether to append to or replace --output-csv (default: append)",
    )
    parser.add_argument(
        "--experiment-workers",
        type=int,
        default=1,
        help="Worker processes for running multiple experiments concurrently "
        "(default: 1, serial). Overlaps per-run data loading.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    args.experiments = _resolve_experiments(args)

    experiments = list(args.experiments)
    exp_workers = min(max(int(args.experiment_workers), 1), len(experiments))

    if exp_workers > 1:
        import concurrent.futures as _cf

        logger.info(
            "Running %d experiments across %d worker process(es)",
            len(experiments), exp_workers,
        )
        results: dict[str, pd.DataFrame] = {}
        with _cf.ProcessPoolExecutor(max_workers=exp_workers) as pool:
            futures = {
                pool.submit(_scalar_bands_one_experiment, args, exp): exp
                for exp in experiments
            }
            for fut in _cf.as_completed(futures):
                exp = futures[fut]
                try:
                    results[exp] = fut.result()
                except Exception:  # noqa: BLE001 - isolate one bad run from the batch
                    logger.warning("Skipping %s: band diagnostics failed", exp, exc_info=True)
        frames = [results[exp] for exp in experiments if exp in results]
    else:
        frames = []
        for exp in experiments:
            try:
                frames.append(_scalar_bands_one_experiment(args, exp))
            except Exception:  # noqa: BLE001 - isolate one bad run from the batch
                logger.warning("Skipping %s: band diagnostics failed", exp, exc_info=True)

    if not frames:
        raise SystemExit("Band diagnostics failed for every experiment; no CSV written.")

    output = Path(args.output_csv)
    combined = pd.concat(frames, ignore_index=True)
    action, previous_rows, new_rows = _write_csv(combined, output, mode=args.csv_mode)
    logger.info(
        "%s %s bands CSV: %s (%d new rows, %d previous rows)",
        action.capitalize(),
        args.component,
        output,
        new_rows,
        previous_rows,
    )


if __name__ == "__main__":
    main()
