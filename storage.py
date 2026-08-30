"""Persistence of raw traces and compact BO history."""

import csv
import dataclasses
import json

import numpy as np

from config import ExperimentConfig
from measurement import Measurement


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
