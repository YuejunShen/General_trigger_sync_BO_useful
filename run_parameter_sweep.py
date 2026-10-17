"""Sweep the number of voltage-protocol discretization points.

Edit PROTOCOL_BASES, DISCRETIZATION_POINTS and REPEATS, then run this file.
Each run gets an independent optimizer and output folder; runs of each basis
go in their own <SWEEP_NAME>/<basis>/ subfolder. The Stop safely button ends
the current run after its active measurement and prevents the next run starting.
"""

import dataclasses

from config import ExperimentConfig
from plot_sweep_convergence import plot_convergence
from run_experiment import run_once
from storage import RUN_LOG_PATH


# "points": linearly interpolated voltages; "bernstein": Bernstein-polynomial
# coefficients (see WaveformConfig.protocol_basis).
PROTOCOL_BASES = ["points", "bernstein"]
DISCRETIZATION_POINTS = [4, 6, 8, 10, 12, 14]
REPEATS = 3
SWEEP_NAME = "protocol_points_scan"


def build_configs():
    """Create each config immediately before its run so timestamps stay fresh."""
    sweep_folder = ExperimentConfig().data_mother_path / SWEEP_NAME
    for repeat in range(1, REPEATS + 1):
        for basis in PROTOCOL_BASES:
            for points in DISCRETIZATION_POINTS:
                cfg = ExperimentConfig()
                waveform = dataclasses.replace(cfg.waveform, protocol_basis=basis,
                                               protocol_points=points)
                tag = waveform.protocol_label
                if REPEATS > 1:
                    tag += f"_rep{repeat}"
                cfg = dataclasses.replace(
                    cfg,
                    data_mother_path=sweep_folder / basis,
                    waveform=waveform,
                    sweep_tag=tag,
                )
                overrides = {"waveform.protocol_basis": basis,
                             "waveform.protocol_points": points}
                yield cfg, overrides, repeat


def main() -> None:
    if REPEATS < 1:
        raise ValueError("REPEATS must be at least 1")
    if not DISCRETIZATION_POINTS:
        raise ValueError("DISCRETIZATION_POINTS must not be empty")
    if not PROTOCOL_BASES:
        raise ValueError("PROTOCOL_BASES must not be empty")

    sweep_folder = ExperimentConfig().data_mother_path / SWEEP_NAME
    total = len(PROTOCOL_BASES) * len(DISCRETIZATION_POINTS) * REPEATS
    print(f"Sweep: {total} run(s) total over bases {PROTOCOL_BASES}. Output folder: {sweep_folder}")
    for index, (cfg, overrides, repeat) in enumerate(build_configs(), start=1):
        print(f"\n=== Run {index}/{total}: {cfg.sweep_tag} ===")
        stopped = run_once(
            cfg,
            sweep_label=cfg.sweep_tag,
            sweep_overrides=overrides,
            repeat_index=repeat,
            block_on_finish=False,
        )
        if stopped:
            print(f"\nStop requested during run {index}/{total} -- ending sweep early.")
            break

    try:
        plot_path = plot_convergence(RUN_LOG_PATH, sweep_folder=sweep_folder)
        print(f"\nConvergence plot saved to {plot_path}")
    except Exception as exc:
        print(f"\nCould not generate convergence plot: {exc}")


if __name__ == "__main__":
    main()
