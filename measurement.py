"""Upload one voltage protocol and return calibrated, protocol-aligned traces."""

from dataclasses import dataclass
import time

import numpy as np

from config import ExperimentConfig
from instruments import ArbGenerator, configured_scope, read_scope_channel
from waveform import build_arb_waveform


@dataclass(frozen=True)
class Measurement:
    time_s: np.ndarray
    current_a: np.ndarray
    voltage_v: np.ndarray
    applied_control_voltages_v: np.ndarray


def apply_voltage_and_measure(control_voltages_v: np.ndarray, cfg: ExperimentConfig) -> Measurement:
    arb = build_arb_waveform(control_voltages_v, cfg.waveform)
    if cfg.scope_periods < 1:
        raise ValueError("scope_periods must be at least 1")
    capture_duration_s = cfg.scope_periods * arb.duration_s
    generator = ArbGenerator(cfg)
    generator.upload(arb)
    try:
        with configured_scope(cfg, capture_duration_s) as scope:
            # Arm first; the ARB sync/trigger output marks protocol t=0.
            channels = (cfg.instruments.current_channel, cfg.instruments.voltage_channel,
                        cfg.instruments.trigger_channel)
            scope.write(":DIGITIZE " + ",".join(f"CHANNEL{ch}" for ch in channels))
            generator.output(True)
            time.sleep(cfg.output_on_wait_s)
            scope.query("*OPC?")
            time_i, raw_i = read_scope_channel(scope, cfg.instruments.current_channel)
            time_v, raw_v = read_scope_channel(scope, cfg.instruments.voltage_channel)
            time_trigger, trigger_v = read_scope_channel(scope, cfg.instruments.trigger_channel)
    finally:
        generator.output(False)

    if not (time_i.size == time_v.size == time_trigger.size):
        raise RuntimeError("Scope returned unequal current, voltage, and trigger lengths")
    current = (raw_i * cfg.instruments.current_termination_correction
               * cfg.instruments.current_gain_a_per_v)
    voltage = raw_v * cfg.instruments.voltage_scale_factor
    # Locate the measured positive trigger-out edge; it is protocol t=0.
    level = cfg.instruments.trigger_level_v
    crossings = np.flatnonzero((trigger_v[:-1] < level) & (trigger_v[1:] >= level))
    if crossings.size:
        index = int(crossings[0])
        dv = trigger_v[index + 1] - trigger_v[index]
        fraction = (level - trigger_v[index]) / dv
        crossing_s = time_trigger[index] + fraction * (time_trigger[index + 1] - time_trigger[index])
    else:
        above = np.flatnonzero(trigger_v >= level)
        if above.size == 0:
            raise RuntimeError("Could not locate the channel-3 trigger-out edge")
        crossing_s = time_trigger[int(above[0])]
    relative_time = time_i - crossing_s
    mask = (relative_time >= 0.0) & (relative_time <= cfg.waveform.protocol_duration_s)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("The scope capture does not contain the BO protocol interval")
    return Measurement(relative_time[mask], current[mask], voltage[mask], np.asarray(control_voltages_v).copy())
