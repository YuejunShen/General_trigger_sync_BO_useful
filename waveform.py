"""Pure functions for constructing the physical ARB waveform."""

from dataclasses import dataclass

import numpy as np

from config import WaveformConfig


@dataclass(frozen=True)
class ArbWaveform:
    voltage_v: np.ndarray
    sample_rate_hz: float
    protocol_start_s: float

    @property
    def duration_s(self) -> float:
        return self.voltage_v.size / self.sample_rate_hz


def _constant(value: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    count = int(round(duration_s * sample_rate_hz))
    return np.full(max(0, count), value, dtype=float)


def _ramp(start: float, stop: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    count = int(round(duration_s * sample_rate_hz))
    if count <= 0:
        return np.empty(0, dtype=float)
    return np.linspace(start, stop, count, endpoint=False, dtype=float)


def build_arb_waveform(control_voltages_v: np.ndarray, cfg: WaveformConfig) -> ArbWaveform:
    """Build the BO protocol followed by a gradual ramp-down and zero hold."""
    controls = np.asarray(control_voltages_v, dtype=float)
    if controls.shape != (cfg.protocol_points,):
        raise ValueError(f"Expected {cfg.protocol_points} voltage points, got {controls.shape}")
    if not np.all(np.isfinite(controls)):
        raise ValueError("Voltage points must be finite")
    allowed_minimum = min(cfg.linear_minimum_voltage_v, cfg.minimum_voltage_v)
    allowed_maximum = max(cfg.linear_maximum_voltage_v, cfg.maximum_voltage_v)
    if np.any((controls < allowed_minimum) | (controls > allowed_maximum)):
        raise ValueError("A protocol voltage point is outside the configured linear/BO ranges")
    if cfg.protocol_ramp_down_s < 0:
        raise ValueError("protocol_ramp_down_s cannot be negative")
    neutral_durations = (cfg.neutralization_fall_time_s,
                         cfg.neutralization_hold_time_s,
                         cfg.neutralization_rise_time_s)
    if cfg.neutralization_voltage_v >= 0 or any(value < 0 for value in neutral_durations):
        raise ValueError("Neutralization voltage must be negative and durations nonnegative")

    protocol_count = max(2, int(round(cfg.protocol_duration_s * cfg.sample_rate_hz)))
    sample_t = np.arange(protocol_count) / cfg.sample_rate_hz
    control_t = np.linspace(sample_t[0], sample_t[-1], cfg.protocol_points)
    protocol = np.interp(sample_t, control_t, controls)
    protocol_ramp_down = _ramp(
        controls[-1], 0.0, cfg.protocol_ramp_down_s, cfg.sample_rate_hz
    )
    neutral_fall = _ramp(0.0, cfg.neutralization_voltage_v,
                         cfg.neutralization_fall_time_s, cfg.sample_rate_hz)
    neutral_hold = _constant(cfg.neutralization_voltage_v,
                             cfg.neutralization_hold_time_s, cfg.sample_rate_hz)
    neutral_rise = _ramp(cfg.neutralization_voltage_v, 0.0,
                         cfg.neutralization_rise_time_s, cfg.sample_rate_hz)
    tail = _constant(0.0, cfg.final_zero_duration_s, cfg.sample_rate_hz)
    voltage = np.concatenate((protocol, protocol_ramp_down, neutral_fall,
                              neutral_hold, neutral_rise, tail))
    if voltage.size > cfg.maximum_arb_points:
        raise ValueError(f"ARB waveform has {voltage.size} points; limit is {cfg.maximum_arb_points}")
    return ArbWaveform(
        voltage_v=voltage,
        sample_rate_hz=cfg.sample_rate_hz,
        protocol_start_s=0.0,
    )


def normalize_for_generator(voltage_v: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return normalized samples, Vpp, and offset without changing physical voltage."""
    low, high = float(np.min(voltage_v)), float(np.max(voltage_v))
    if np.isclose(low, high):
        return np.zeros_like(voltage_v), 0.02, low
    vpp = high - low
    offset = (high + low) / 2.0
    normalized = 2.0 * (voltage_v - low) / vpp - 1.0
    return normalized, vpp, offset
