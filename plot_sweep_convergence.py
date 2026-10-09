"""Plot best feasible work versus protocol discretization-point count."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def work_ratio_for_run(output_directory: Path) -> float | None:
    history_files = list(output_directory.glob("bo_history_*.csv"))
    if not history_files:
        return None
    history = pd.read_csv(history_files[0])
    if history.empty or "VI_integral" not in history:
        return None
    reference = history.sort_values("evaluation")["VI_integral"].iloc[0]
    if reference == 0:
        return None
    feasible = history[history["feasible"].astype(str).str.lower().eq("true")]
    if feasible.empty:
        return None
    return feasible["VI_integral"].min() / reference


def plot_convergence(run_log_path: Path, sweep_folder: Path) -> Path:
    log = pd.read_csv(run_log_path)
    resolved_folder = Path(sweep_folder).resolve()
    log = log[log["output_directory"].apply(
        lambda value: Path(value).resolve().parent == resolved_folder
    )]
    if log.empty:
        raise ValueError("No matching completed runs in the run log")

    log = log.assign(work_ratio=[work_ratio_for_run(Path(path))
                                 for path in log["output_directory"]])
    log = log.dropna(subset=["work_ratio"])
    if log.empty:
        raise ValueError("No matching runs contain feasible history data")

    grouped = log.groupby("protocol_points")["work_ratio"]
    x = grouped.mean().index.to_numpy(float)
    means = grouped.mean().to_numpy(float)
    stds = grouped.std(ddof=0).fillna(0.0).to_numpy(float)
    order = np.argsort(x)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.errorbar(x[order], means[order], yerr=stds[order], marker="o", capsize=4)
    axis.axhline(1.0, color="0.6", linestyle="--", label="linear-protocol baseline")
    axis.set(xlabel="Number of protocol discretization points",
             ylabel="Best feasible work / linear-protocol work",
             title="BO convergence vs. protocol discretization")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    output_path = resolved_folder / "sweep_work_ratio_convergence.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
