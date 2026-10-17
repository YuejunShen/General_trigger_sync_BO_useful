"""Persistence of raw traces and compact BO history."""

import csv
import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from measurement import Measurement


# Kept separate from the PUND run log because the two experiment types record
# different parameter columns.
RUN_LOG_PATH = Path(r"D:\Ferroelectric_BTO_optimization\general_run_log.csv")
RUN_LOG_COLUMNS = ["logged_at", "run_timestamp", "circuit_type",
                   "protocol_basis", "protocol_points", "initial_random_measurements", "bo_steps",
                   "q_lower_relative_tolerance", "sweep_label", "repeat_index",
                   "sweep_overrides_json", "output_directory"]
# Value given to a column for runs logged before that column existed.
_RUN_LOG_LEGACY_DEFAULTS = {"protocol_basis": "points"}


def _upgrade_run_log() -> list[str]:
    """Rewrite an older run log so it has every current column; return its header."""
    with RUN_LOG_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        old_columns = reader.fieldnames or []
        rows = list(reader)
    columns = RUN_LOG_COLUMNS + [name for name in old_columns if name not in RUN_LOG_COLUMNS]
    if old_columns == columns:
        return columns
    for row in rows:
        for name in columns:
            if name not in old_columns:
                row[name] = _RUN_LOG_LEGACY_DEFAULTS.get(name, "")
    with RUN_LOG_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return columns


def append_run_log(cfg: ExperimentConfig, sweep_label: str = "",
                   sweep_overrides: dict | None = None,
                   repeat_index: int | None = None) -> None:
    """Record one single or swept run and the protocol discretization it used."""
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = _upgrade_run_log() if RUN_LOG_PATH.exists() else None
    row = {
        "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_timestamp": cfg.run_timestamp,
        "circuit_type": cfg.circuit.circuit_type,
        "protocol_basis": cfg.waveform.protocol_basis,
        "protocol_points": cfg.waveform.protocol_points,
        "initial_random_measurements": cfg.bo.initial_random_measurements,
        "bo_steps": cfg.bo.bo_steps,
        "q_lower_relative_tolerance": cfg.bo.q_lower_relative_tolerance,
        "sweep_label": sweep_label,
        "repeat_index": repeat_index if repeat_index is not None else "",
        "sweep_overrides_json": json.dumps(sweep_overrides or {}),
        "output_directory": str(cfg.output_directory),
    }
    with RUN_LOG_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or RUN_LOG_COLUMNS, restval="")
        if columns is None:
            writer.writeheader()
        writer.writerow(row)


def save_run_config(cfg: ExperimentConfig) -> None:
    """Write every scan parameter (instrument, waveform, BO, circuit settings) to a log file."""
    cfg.output_directory.mkdir(parents=True, exist_ok=True)
    config_path = cfg.output_directory / f"run_config_{cfg.run_timestamp}.json"
    payload = dataclasses.asdict(cfg)
    payload["output_directory"] = str(cfg.output_directory)
    payload["history_file"] = str(cfg.history_file)
    with config_path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_measurement(measurement: Measurement, iteration: int, result: dict, cfg: ExperimentConfig) -> None:
    cfg.output_directory.mkdir(parents=True, exist_ok=True)
    trace_path = cfg.output_directory / (
        f"trace_{cfg.run_timestamp}_iteration_{iteration:03d}.csv"
    )
    with trace_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protocol_time_s", "current_a", "measured_voltage_v"])
        writer.writerows(zip(measurement.time_s, measurement.current_a, measurement.voltage_v))
    new_file = not cfg.history_file.exists()
    with cfg.history_file.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["evaluation", "trace_file", *[f"V{i}" for i in range(measurement.applied_control_voltages_v.size)],
                             "VI_integral", "VI_integral_NORM", "Q_tau", "Q_tau_NORM",
                             "Q_target", "Q_lower_bound", "Q_violation", "Q_lower", "Q_lower_NORM",
                             "Q_SCALE", "WORK_SCALE",
                             "feasible", "phase", "phase_evaluation"])
        writer.writerow([iteration, trace_path.name, *measurement.applied_control_voltages_v,
                         result["VI_integral"], result["VI_integral_NORM"],
                         result["Q_tau"], result["Q_tau_NORM"], result["Q_target"],
                         result["Q_lower_bound"], result["Q_violation"], result["Q_lower"], result["Q_lower_NORM"],
                         result["Q_SCALE"], result["WORK_SCALE"],
                         result["feasible"], result["phase"],
                         result["phase_evaluation"]])
