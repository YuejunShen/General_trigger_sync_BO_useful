"""Configuration for the sequential voltage-optimization experiment."""

from dataclasses import dataclass, field
import datetime as dt
import time
from pathlib import Path


@dataclass(frozen=True)
class InstrumentConfig:
    function_generator_address: str = "USB0::0x0957::0x2707::MY57301577::INSTR"
    oscilloscope_address: str = "USB0::0x0957::0x9005::MY50140103::INSTR"
    function_generator_timeout_ms: int = 100_000
    oscilloscope_timeout_ms: int = 100_000
    current_channel: int = 1
    voltage_channel: int = 2
    trigger_channel: int = 3
    # True: channel 1 measures a current-amplifier output, so convert using
    # current_gain_a_per_v and current_termination_correction. False: channel 1
    # is connected directly across the scope's 50-ohm input and I = V_scope/R.
    use_current_amplifier: bool = True
    current_gain_a_per_v: float = 1e-4
    current_input_impedance_ohm: float = 50.0
    current_termination_correction: float = 2.0
    voltage_scale_factor: float = 1.0
    current_scale_v_per_div: float = 0.3
    voltage_scale_v_per_div: float = 0.5
    current_offset_v: float = 0.0
    voltage_offset_v: float = 0.0
    trigger_scale_v_per_div: float = 1.0
    trigger_offset_v: float = 0.0
    trigger_level_v: float = 0.5

    def __post_init__(self):
        if self.current_input_impedance_ohm <= 0.0:
            raise ValueError("current_input_impedance_ohm must be positive")
        if self.current_gain_a_per_v <= 0.0:
            raise ValueError("current_gain_a_per_v must be positive")
        if self.current_termination_correction <= 0.0:
            raise ValueError("current_termination_correction must be positive")

    @property
    def effective_current_gain_a_per_v(self) -> float:
        """Conversion from the scope's measured volts to current in amperes."""
        if self.use_current_amplifier:
            return self.current_termination_correction * self.current_gain_a_per_v
        return 1.0 / self.current_input_impedance_ohm


@dataclass(frozen=True)
class WaveformConfig:
    sample_rate_hz: float = 1e3
    protocol_duration_s: float = 0.1
    protocol_points: int = 14
    # How the protocol_points BO variables define V(t) over the protocol:
    # "points"    -- voltages at evenly spaced times, linearly interpolated.
    # "bernstein" -- coefficients of a degree (protocol_points - 1) Bernstein
    #                polynomial. V(t) always stays between the smallest and
    #                largest coefficient, so the BO voltage bounds hold without
    #                clipping; V(0) is the first and V(end) the last coefficient.
    protocol_basis: str = "bernstein"
    protocol_ramp_down_s: float = 0.1
    neutralization_voltage_v: float = -1
    neutralization_fall_time_s: float = 0.1
    neutralization_hold_time_s: float = 0.1
    neutralization_rise_time_s: float = 0.1
    final_zero_duration_s: float = 1
    linear_minimum_voltage_v: float = 0.0
    linear_maximum_voltage_v: float = 1.0
    minimum_voltage_v: float = 0.0  # BO lower bound
    maximum_voltage_v: float = 1.3
    maximum_arb_points: int = 1_000_000

    def __post_init__(self):
        if self.protocol_points < 2:
            raise ValueError("protocol_points must be at least 2")
        if self.protocol_basis not in ("points", "bernstein"):
            raise ValueError(f"Unknown protocol_basis {self.protocol_basis!r}")

    @property
    def protocol_label(self) -> str:
        """Short tag such as "points10" or "bernstein10" for folder names."""
        return f"{self.protocol_basis}{self.protocol_points}"


@dataclass(frozen=True)
class BOConfig:
    initial_random_measurements: int = 30
    bo_steps: int = 300
    # Shifts the BO feasibility boundary relative to Q_TARGET (the first,
    # linear-protocol evaluation's Q_tau), as a fraction of |Q_TARGET|:
    # negative LOOSENS the bound (Q_tau may fall below Q_TARGET by that
    # fraction and still be searched as feasible), positive TIGHTENS it,
    # 0.0 means the search boundary sits exactly at Q_TARGET. Does not
    # affect the "feasible" column, which always reports Q_tau >= Q_TARGET.
    q_lower_relative_tolerance: float = 4e-3
    random_seed: int = field(default_factory=lambda: time.time_ns() % (2**32))
    n_restarts: int = 5
    max_iter: int = 100
    # None avoids Botorch's Windows threadpoolctl timeout wrapper.
    max_time_s: float | None = None

    @property
    def total_measurements(self) -> int:
        return 1 + self.initial_random_measurements + self.bo_steps


@dataclass(frozen=True)
class CircuitConfig:
    # "series_rc" | "series_r1_parallel_r2_c" | "arb_circuit". "arb_circuit" is
    # for a circuit with no known equation of motion: circuits.py has nothing
    # to simulate or solve analytically for it, so make_evolution_video.py and
    # ControlPanel simply skip the simulated-current/analytical-protocol
    # comparisons and fall back to the measured trace alone.
    circuit_type: str = "series_rc"
    resistance_ohm: float = 10e3             # series_rc
    capacitance_f: float = 10e-6           # series_rc, series_r1_parallel_r2_c
    r1_ohm: float = 10e3                   # series_r1_parallel_r2_c
    r2_ohm: float = 10e3                   # series_r1_parallel_r2_c
    initial_charge_c: float = 0.0
    show_theoretical_current: bool = True

    @property
    def params(self) -> dict:
        """Keyword args expected by circuits.simulate_circuit / analytical_optimal_protocol."""
        if self.circuit_type == "series_rc":
            return {"resistance_ohm": self.resistance_ohm, "capacitance_f": self.capacitance_f}
        if self.circuit_type == "series_r1_parallel_r2_c":
            return {"r1_ohm": self.r1_ohm, "r2_ohm": self.r2_ohm, "capacitance_f": self.capacitance_f}
        if self.circuit_type == "arb_circuit":
            return {}
        raise ValueError(f"Unknown circuit_type {self.circuit_type!r}")

    @property
    def folder_label(self) -> str:
        if self.circuit_type == "series_rc":
            return f"R={self.resistance_ohm:.12g}_C={self.capacitance_f:.12g}"
        if self.circuit_type == "series_r1_parallel_r2_c":
            return f"R1={self.r1_ohm:.12g}_R2={self.r2_ohm:.12g}_C={self.capacitance_f:.12g}"
        if self.circuit_type == "arb_circuit":
            return "arb_circuit"
        raise ValueError(f"Unknown circuit_type {self.circuit_type!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    instruments: InstrumentConfig = field(default_factory=InstrumentConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    run_timestamp: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    data_mother_path: Path = field(
        default_factory=lambda: Path(r"D:\Ferroelectric_BTO_optimization")
        / dt.datetime.now().strftime("%Y%m%d")
    )
    scope_time_step_s: float = 20e-6
    scope_periods: int = 2
    output_on_wait_s: float = 0.1
    enable_control_panel: bool = True
    # Set by run_parameter_sweep.py so every swept run has a distinct,
    # self-documenting output folder. Empty for an ordinary single run, which
    # is then tagged with waveform.protocol_label instead.
    sweep_tag: str = ""

    @property
    def output_directory(self) -> Path:
        # HHMM from run_timestamp so repeat runs (same circuit, same day) get their own folder.
        time_of_day = self.run_timestamp.split("_")[1][:4]
        suffix = self.sweep_tag or self.waveform.protocol_label
        return self.data_mother_path / f"BO_{self.circuit.folder_label}_{time_of_day}_{suffix}"

    @property
    def history_file(self) -> Path:
        return self.output_directory / f"bo_history_{self.run_timestamp}.csv"
