"""Create an MP4 of an experimental BO run from its saved CSV files."""

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import circuits
from config import ExperimentConfig


# User-editable folder containing bo_history_*.csv and trace_*.csv files.
DATA_FOLDER = Path(r"D:\Ferroelectric_BTO_optimization\20260806\BO_R1=10000_R2=10000_C=1e-05_1912")

# Marker shape encodes which phase an evaluation belongs to; color encodes
# feasibility (independent of phase) -- see the work-vs-reference panel.
_PHASE_MARKERS = {"linear protocol": "s", "random initial": "x", "BO optimization": "o"}
_FEASIBLE_COLORS = {True: "tab:green", False: "tab:red"}
_WORK_PANEL_LEGEND_HANDLES = [
    Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="k", label="linear protocol"),
    Line2D([0], [0], marker="x", linestyle="None", color="k", label="random initial"),
    Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="k", label="BO optimization"),
    Line2D([0], [0], marker="o", linestyle="None", color=_FEASIBLE_COLORS[True], label="feasible"),
    Line2D([0], [0], marker="o", linestyle="None", color=_FEASIBLE_COLORS[False], label="infeasible"),
]


def latest_history(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Data folder does not exist: {directory}")
    files = list(directory.glob("bo_history_*.csv"))
    legacy_history = directory / "bo_history.csv"
    if legacy_history.exists():
        files.append(legacy_history)
    if not files:
        raise FileNotFoundError(f"No BO history CSV found in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_run_config(history_path: Path) -> dict | None:
    """Load the run_config_<timestamp>.json written alongside bo_history_<timestamp>.csv,
    if one exists (older runs predate that log and simply won't have an analytical curve)."""
    if not history_path.stem.startswith("bo_history_"):
        return None
    run_timestamp = history_path.stem.removeprefix("bo_history_")
    config_path = history_path.parent / f"run_config_{run_timestamp}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


def load_run(history_path: Path):
    history = pd.read_csv(history_path).sort_values("evaluation").reset_index(drop=True)
    if history.empty:
        raise ValueError("History is empty")
    voltage_columns = sorted(
        (name for name in history if name.startswith("V") and name[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    if not voltage_columns:
        raise ValueError("History contains no voltage control-point columns")
    if "phase" not in history:
        history["phase"] = "BO optimization"
        history.loc[0, "phase"] = "linear protocol"
    if "Q_lower_bound" not in history:
        history["Q_lower_bound"] = history["Q_target"]
    traces = []
    for _, row in history.iterrows():
        if "trace_file" in history:
            path = history_path.parent / str(row["trace_file"])
        else:
            pattern = f"iteration_{int(row['evaluation']):03d}_*.csv"
            matches = list(history_path.parent.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No legacy trace matching {pattern}")
            path = max(matches, key=lambda candidate: candidate.stat().st_mtime)
        if not path.exists():
            raise FileNotFoundError(f"Missing trace referenced by history: {path}")
        traces.append(pd.read_csv(path))
    return history, voltage_columns, traces


def make_video(history_path: Path, output_path: Path, fps: int = 6) -> Path:
    history, voltage_columns, traces = load_run(history_path)
    feasible = history["feasible"].astype(str).str.lower().eq("true")
    history = history.assign(feasible=feasible)

    reference_trace = traces[0]
    reference_time = reference_trace["protocol_time_s"].to_numpy(float)
    reference_v = reference_trace["measured_voltage_v"].to_numpy(float)
    reference_current = reference_trace["current_a"].to_numpy(float)

    run_config = load_run_config(history_path)
    circuit_type = None
    circuit_params = None
    q_initial = 0.0
    reference_vi = float(history.iloc[0]["VI_integral"])

    q_target = float(history.iloc[0]["Q_target"])
    q_lower_bound = float(history.iloc[0]["Q_lower_bound"])

    analytical_v = None
    analytical_q = None
    analytical_excess = None
    reference_charge = None
    if run_config is not None:
        circuit_config = run_config["circuit"]
        # Older run_config logs predate circuit_type and were always series_rc.
        circuit_type = circuit_config.get("circuit_type", "series_rc")
        circuit_params = circuit_config
        q_initial = float(circuit_config["initial_charge_c"])
        analytical = circuits.analytical_optimal_protocol(
            circuit_type, circuit_params, reference_time, q_initial, q_lower_bound
        )
        if analytical is not None:
            analytical_v, analytical_q, analytical_vi = analytical
            analytical_excess = (analytical_vi - reference_vi) / abs(reference_vi)
        else:
            print(f"Circuit type {circuit_type!r} has no analytical optimal-protocol solver; "
                  "skipping that comparison curve.")
        # Q_C(t) is always recoverable from the measured trace (see
        # circuits.charge_trajectory_from_trace: algebraic via Ohm's law
        # across R1 for series_r1_parallel_r2_c, cumulative integral of
        # I_tot for series_rc/arb_circuit), so the Charge-on-C panel is
        # shown alongside the Current panel for every circuit type, not
        # just series_r1_parallel_r2_c.
        reference_charge = circuits.charge_trajectory_from_trace(
            circuit_type, circuit_params, reference_time, reference_current, reference_v, q_initial
        )
    else:
        print(f"No run_config_*.json found next to {history_path.name}; "
              "skipping the analytical-optimal-protocol, simulated-current, and charge-on-C panels.")

    excess_work = (history["VI_integral"] - reference_vi) / abs(reference_vi)

    q_all = history["Q_tau"].to_numpy(float)
    q_pad = max(0.15 * abs(q_target), 5.0 * abs(q_lower_bound - q_target), 1e-12)
    q_axis_low = min(q_all.min(), q_lower_bound) - q_pad
    q_axis_high = max(q_all.max(), q_target) + q_pad

    excess_bounds = [excess_work.min(), excess_work.max()] + ([analytical_excess] if analytical_excess is not None else [])
    excess_axis_low, excess_axis_high = min(excess_bounds), max(excess_bounds)
    excess_pad = 0.05 * max(excess_axis_high - excess_axis_low, 1e-12)

    figure, axes = plt.subplots(2, 3, figsize=(17, 8))
    ax_v, ax_current, ax_charge = axes[0]
    ax_bar, ax_work, ax_unused = axes[1]
    ax_unused.set_visible(False)
    show_charge_panel = circuit_type is not None

    def update(frame):
        row = history.iloc[frame]
        trace = traces[frame]
        time_s = trace["protocol_time_s"].to_numpy(float)
        voltage_v = trace["measured_voltage_v"].to_numpy(float)
        current_a = trace["current_a"].to_numpy(float)
        controls = row[voltage_columns].to_numpy(float)
        control_t = np.linspace(time_s[0], time_s[-1], controls.size)
        is_feasible = bool(row["feasible"])
        bar_color = _FEASIBLE_COLORS[is_feasible]

        axes_to_clear = (ax_v, ax_current, ax_bar, ax_work) + ((ax_charge,) if show_charge_panel else ())
        for axis in axes_to_clear:
            axis.clear()

        ax_v.plot(reference_time, reference_v, "k--", linewidth=1.2, label="linear protocol V(t)")
        if analytical_v is not None:
            ax_v.plot(reference_time, analytical_v, color="tab:purple", linestyle="--", linewidth=1.8,
                      label="analytical optimal V(t)")
        ax_v.plot(time_s, voltage_v, color="tab:blue", linewidth=2, label="measured V(t)")
        ax_v.scatter(control_t, controls, color="tab:blue", s=24, zorder=4)
        ax_v.set(title="Voltage protocol", xlabel="Protocol time (s)", ylabel="Voltage (V)")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="upper left", fontsize=8)

        theory = circuits.simulate_circuit(circuit_type, circuit_params, time_s, voltage_v, q_initial) \
            if circuit_type is not None else None

        ax_current.plot(time_s, current_a, color="tab:green", linewidth=1.8, label="measured I(t)")
        if theory is not None:
            ax_current.plot(time_s, theory.current_a, "--", color="tab:orange", linewidth=1.8,
                             label="simulated I(t) (circuit model)")
        ax_current.set(title="Current", xlabel="Protocol time (s)", ylabel="Current (A)")
        ax_current.grid(True, alpha=0.3)
        ax_current.legend(loc="upper right", fontsize=8)

        if show_charge_panel:
            # Q_C(t) recovered from the measured trace -- algebraic via Ohm's
            # law across R1 for series_r1_parallel_r2_c, cumulative integral
            # of I_tot for series_rc/arb_circuit (see
            # circuits.charge_trajectory_from_trace).
            measured_charge = circuits.charge_trajectory_from_trace(
                circuit_type, circuit_params, time_s, current_a, voltage_v, q_initial
            )
            ax_charge.plot(reference_time, reference_charge, "k--", linewidth=1.2, label="linear protocol Q_C(t)")
            if analytical_q is not None:
                ax_charge.plot(reference_time, analytical_q, color="tab:purple", linestyle="--", linewidth=1.8,
                                label="analytical optimal Q_C(t)")
            if theory is not None:
                ax_charge.plot(time_s, theory.charge_c, "--", color="tab:orange", linewidth=1.8,
                                label="simulated Q_C(t) (circuit model)")
            ax_charge.plot(time_s, measured_charge, color="tab:blue", linewidth=2, label="measured Q_C(t)")
            ax_charge.axhline(q_target, color="k", linestyle=":", linewidth=1, label="target Q_tau")
            ax_charge.axhline(q_lower_bound, color="tab:red", linestyle="--", linewidth=1.2,
                               label="BO feasibility boundary")
            ax_charge.set(title="Charge on C", xlabel="Protocol time (s)", ylabel="Charge (C)")
            ax_charge.grid(True, alpha=0.3)
            ax_charge.legend(loc="upper right", fontsize=8)

        ax_bar.bar([0], [row["Q_tau"]], width=0.5, color=bar_color, alpha=0.6)
        ax_bar.axhline(q_lower_bound, color="tab:red", linestyle="--", linewidth=1.6, label="BO feasibility boundary")
        ax_bar.axhline(q_target, color="k", linestyle=":", linewidth=1.4, label="target Q_tau")
        ax_bar.set_xlim(-0.8, 0.8)
        ax_bar.set_ylim(q_axis_low, q_axis_high)
        ax_bar.set_xticks([0])
        ax_bar.set_xticklabels(["Q_tau"])
        ax_bar.set_ylabel("Charge (C)")
        ax_bar.set_title("Feasible" if is_feasible else "Constraint violated")
        ax_bar.legend(loc="upper right", fontsize=8)
        ax_bar.grid(True, axis="y", alpha=0.3)

        past = history.iloc[: frame + 1]
        for phase_name, marker in _PHASE_MARKERS.items():
            subset = past[past["phase"] == phase_name]
            if subset.empty:
                continue
            subset_excess = excess_work.loc[subset.index]
            colors = subset["feasible"].map(_FEASIBLE_COLORS)
            ax_work.scatter(subset["evaluation"], subset_excess,
                            s=32 if marker == "x" else 22, alpha=0.85, c=colors, marker=marker)
        ax_work.axhline(0.0, color="k", linestyle="--", linewidth=1)
        work_legend_handles = list(_WORK_PANEL_LEGEND_HANDLES)
        work_legend_handles.append(Line2D([0], [0], color="k", linestyle="--", linewidth=1, label="linear protocol work"))
        if analytical_excess is not None:
            ax_work.axhline(analytical_excess, color="tab:purple", linestyle="--", linewidth=1.4)
            work_legend_handles.append(
                Line2D([0], [0], color="tab:purple", linestyle="--", linewidth=1.4, label="analytical optimal work")
            )
        ax_work.set_xlim(-1, len(history))
        ax_work.set_ylim(excess_axis_low - excess_pad, excess_axis_high + excess_pad)
        ax_work.set(xlabel="Evaluation", ylabel="(work - linear protocol work) / linear protocol work",
                    title="Work vs. linear protocol")
        ax_work.grid(True, alpha=0.3)
        ax_work.legend(handles=work_legend_handles, loc="upper right", fontsize=10, ncol=2)

        figure.suptitle(
            f"Evaluation {int(row['evaluation'])} | {row['phase']} | "
            f"feasible={is_feasible}",
            fontsize=14,
        )
        figure.tight_layout(pad=2.0)
        return []

    movie = animation.FuncAnimation(figure, update, frames=len(history),
                                    interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f"Unsupported video extension {suffix!r}; use .mp4 or .gif.")
    movie.save(output_path, writer=writer)
    plt.close(figure)
    return output_path


def default_output_path(history_path: Path) -> Path:
    if history_path.stem.startswith("bo_history_"):
        default_name = history_path.stem.replace("bo_history_", "bo_evolution_", 1) + ".mp4"
    else:
        default_name = "bo_evolution.mp4"
    return history_path.with_name(default_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--data-folder", type=Path, help="Folder containing all run CSV files")
    selection.add_argument("--history", type=Path, help="Exact timestamped bo_history CSV")
    parser.add_argument("--output", type=Path, help="Output video path (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    cfg = ExperimentConfig()
    data_folder = args.data_folder or DATA_FOLDER or cfg.output_directory
    history_path = args.history or latest_history(data_folder)
    output_path = args.output or default_output_path(history_path)
    result = make_video(history_path.resolve(), output_path.resolve(), args.fps)
    print(f"Evolution video saved to {result}")


if __name__ == "__main__":
    main()
