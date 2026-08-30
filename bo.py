"""Xopt constrained-EI optimizer adapted from the reference notebook."""

from dataclasses import dataclass, field
from contextlib import nullcontext
import sys
import numpy as np
import pandas as pd
from packaging.version import Version

import circuits
from config import BOConfig, CircuitConfig, WaveformConfig

try:
    import xopt
    from xopt.generators.bayesian import ExpectedImprovementGenerator
    from xopt.vocs import VOCS
except ImportError:
    xopt = None
    ExpectedImprovementGenerator = None
    VOCS = None


if sys.platform == "win32" and xopt is not None:
    # BoTorch always asks threadpoolctl to enumerate loaded BLAS DLLs before
    # fitting. That Windows API call is intermittently unavailable on this lab
    # PC. Only disable the thread-count wrapper; optimization itself is unchanged.
    import botorch.optim.utils.timeout as _botorch_timeout

    _botorch_timeout.threadpool_limits = lambda *args, **kwargs: nullcontext()


def trapezoidal_integral(y, x):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def trace_metrics(time_s, current_a, voltage_v, circuit_cfg: CircuitConfig):
    """Return integral(V*I dt) (the work drawn from the source -- always
    correct regardless of topology, since current_a is the measured source
    current) and the final capacitor charge Q_tau. Only the total source
    current is measured (there is no separate capacitor-branch probe), so
    Q_tau's formula depends on circuit_type -- see circuits.charge_from_trace
    for why integral(I dt) is only exactly the capacitor's charge when there
    is no bypass path around it."""
    time_s, current_a, voltage_v = map(
        lambda a: np.asarray(a, dtype=float), (time_s, current_a, voltage_v)
    )
    if not (time_s.size == current_a.size == voltage_v.size and time_s.size >= 2):
        raise ValueError("Time, current, and voltage traces must have equal length >= 2")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError("Trace time must be finite and strictly increasing")
    vi_integral = trapezoidal_integral(voltage_v * current_a, time_s)
    q_tau = circuits.charge_from_trace(circuit_cfg.circuit_type, circuit_cfg.params, time_s, current_a, voltage_v,
                                        circuit_cfg.initial_charge_c)
    return vi_integral, q_tau


@dataclass
class BayesianVoltageOptimizer:
    waveform_cfg: WaveformConfig
    bo_cfg: BOConfig
    circuit_cfg: CircuitConfig
    data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    q_target: float | None = field(default=None, init=False)
    q_lower_bound: float | None = field(default=None, init=False)
    q_scale: float | None = field(default=None, init=False)
    work_scale: float | None = field(default=None, init=False)
    q_lower_norm_threshold: float | None = field(default=None, init=False)
    _generator: object | None = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if ExpectedImprovementGenerator is None:
            raise RuntimeError("Xopt is required: pip install xopt")
        if Version(xopt.__version__) < Version("3.2.1"):
            raise RuntimeError(f"Xopt {xopt.__version__} is unsupported; use .\\run_in_xopt_env.ps1")
        self.input_names = [f"V{i}" for i in range(self.waveform_cfg.protocol_points)]
        self._rng = np.random.default_rng(self.bo_cfg.random_seed)

    def _make_generator(self):
        self.vocs = VOCS(
            variables={name: [self.waveform_cfg.minimum_voltage_v,
                              self.waveform_cfg.maximum_voltage_v]
                       for name in self.input_names},
            objectives={"VI_integral_NORM": "MINIMIZE"},
            constraints={"Q_lower_NORM": ["GREATER_THAN", self.q_lower_norm_threshold]},
            observables=["VI_integral", "Q_tau", "Q_tau_NORM", "Q_violation"],
        )
        generator = ExpectedImprovementGenerator(vocs=self.vocs)
        generator.gp_constructor.use_low_noise_prior = True
        generator.numerical_optimizer.n_restarts = self.bo_cfg.n_restarts
        generator.numerical_optimizer.max_iter = self.bo_cfg.max_iter
        generator.numerical_optimizer.max_time = self.bo_cfg.max_time_s
        self._generator = generator

    def propose(self):
        count = len(self.data)
        low, high = (self.waveform_cfg.minimum_voltage_v,
                     self.waveform_cfg.maximum_voltage_v)
        if count == 0:
            return np.linspace(self.waveform_cfg.linear_minimum_voltage_v,
                               self.waveform_cfg.linear_maximum_voltage_v,
                               self.waveform_cfg.protocol_points)
        if count <= self.bo_cfg.initial_random_measurements:
            return self._rng.uniform(low, high, self.waveform_cfg.protocol_points)
        inputs = self._generator.generate(1)[0]
        return np.array([inputs[name] for name in self.input_names], dtype=float)

    def observe(self, applied_voltages_v, time_s, current_a, measured_voltage_v):
        protocol = np.asarray(applied_voltages_v, dtype=float)
        if protocol.shape != (self.waveform_cfg.protocol_points,):
            raise ValueError("Applied voltage control-point count does not match Xopt")
        vi_integral, q_tau = trace_metrics(time_s, current_a, measured_voltage_v, self.circuit_cfg)
        if self.q_target is None:
            # q_tau of the first (linear-protocol) evaluation defines Q_TARGET.
            # q_lower_relative_tolerance shifts the feasibility boundary relative
            # to Q_TARGET, as a fraction of |Q_TARGET|: negative LOOSENS the
            # bound (Q_tau may fall below Q_TARGET by that fraction and still be
            # searched as feasible by BO), positive TIGHTENS it, 0.0 means the
            # BO search boundary sits exactly at Q_TARGET.
            self.q_target = q_tau
            tolerance = self.bo_cfg.q_lower_relative_tolerance * abs(self.q_target)
            self.q_lower_bound = self.q_target + tolerance
            self.q_scale = abs(self.q_target) if self.q_target != 0.0 else 1.0
            self.work_scale = abs(vi_integral) if vi_integral != 0.0 else 1.0
            self.q_lower_norm_threshold = self.q_lower_bound / self.q_scale
            self._make_generator()
        q_tau_norm = q_tau / self.q_scale
        vi_integral_norm = vi_integral / self.work_scale
        evaluation = len(self.data)
        if evaluation == 0:
            phase, phase_evaluation = "linear protocol", 0
        elif evaluation <= self.bo_cfg.initial_random_measurements:
            phase, phase_evaluation = "random initial", evaluation
        else:
            phase = "BO optimization"
            phase_evaluation = evaluation - self.bo_cfg.initial_random_measurements
        row = dict(zip(self.input_names, protocol))
        row.update({
            "VI_integral": vi_integral,
            "VI_integral_NORM": vi_integral_norm,
            "Q_tau": q_tau,
            "Q_tau_NORM": q_tau_norm,
            "Q_target": self.q_target,
            "Q_lower_bound": self.q_lower_bound,
            "Q_violation": max(self.q_lower_bound - q_tau, 0.0),
            "Q_lower": q_tau,
            "Q_lower_NORM": q_tau_norm,
            "Q_SCALE": self.q_scale,
            "WORK_SCALE": self.work_scale,
            # "feasible" reports whether Q_tau reached the true physical
            # target, independent of q_lower_relative_tolerance -- that
            # tolerance only shifts what the BO constraint itself searches
            # for (q_lower_bound above), it does not redefine what counts as
            # a successful result for reporting.
            "feasible": bool(q_tau >= self.q_target),
            "phase": phase,
            "phase_evaluation": phase_evaluation,
        })
        new_data = pd.DataFrame([row], index=[evaluation])
        self.data = pd.concat([self.data, new_data])
        self._generator.add_data(new_data)
        return row
