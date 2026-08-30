"""Circuit models this rig can drive: an explicit-Euler simulator plus a
closed-form minimum-work solver for each, dispatched by CircuitConfig.circuit_type.

Every built-in circuit here is a single scalar state, the capacitor charge
q(t), integrated under the applied voltage waveform -- consistent with
bo.py's own charge bookkeeping (trace_metrics). "arb_circuit" is the escape
hatch for a circuit with no known equation of motion: simulate_circuit and
analytical_optimal_protocol both return None for it, since there is nothing
to simulate or solve analytically -- only the measured trace exists.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircuitResponse:
    time_s: np.ndarray
    current_a: np.ndarray
    charge_c: np.ndarray


def simulate_series_rc(time_s, applied_voltage_v, resistance_ohm, capacitance_f, initial_charge_c=0.0):
    """V(t) = R * dq/dt + q/C. Match BO_discretize_points_RC.ipynb: Euler Q, then I=dQ/dt."""
    time_s = np.asarray(time_s, dtype=float)
    voltage = np.asarray(applied_voltage_v, dtype=float)
    if time_s.ndim != 1 or voltage.shape != time_s.shape or time_s.size < 2:
        raise ValueError("Time and applied voltage must be equal one-dimensional arrays")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Simulation time must be strictly increasing")
    if resistance_ohm <= 0 or capacitance_f <= 0:
        raise ValueError("Resistance and capacitance must be positive")

    charge = np.zeros_like(time_s)
    charge[0] = float(initial_charge_c)
    for index, dt in enumerate(np.diff(time_s)):
        dq_dt = (voltage[index] - charge[index] / capacitance_f) / resistance_ohm
        charge[index + 1] = charge[index] + dt * dq_dt
    current = np.gradient(charge, time_s)
    return CircuitResponse(time_s.copy(), current, charge)


def simulate_series_r1_parallel_r2_c(time_s, applied_voltage_v, r1_ohm, r2_ohm, capacitance_f, initial_charge_c=0.0):
    """R1 in series with (R2 parallel C); q is the capacitor charge.
    dq/dt = (V - q/C)/R1 - (q/C)/R2. The reported current is the source
    current (V - q/C)/R1, not dq/dt -- part of it bypasses through R2 and
    never reaches the capacitor."""
    time_s = np.asarray(time_s, dtype=float)
    voltage = np.asarray(applied_voltage_v, dtype=float)
    if time_s.ndim != 1 or voltage.shape != time_s.shape or time_s.size < 2:
        raise ValueError("Time and applied voltage must be equal one-dimensional arrays")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("Simulation time must be strictly increasing")
    if r1_ohm <= 0 or r2_ohm <= 0 or capacitance_f <= 0:
        raise ValueError("Resistances and capacitance must be positive")

    charge = np.zeros_like(time_s)
    charge[0] = float(initial_charge_c)
    for index, dt in enumerate(np.diff(time_s)):
        capacitor_voltage = charge[index] / capacitance_f
        dq_dt = (voltage[index] - capacitor_voltage) / r1_ohm - capacitor_voltage / r2_ohm
        charge[index + 1] = charge[index] + dt * dq_dt
    capacitor_voltage = charge / capacitance_f
    source_current = (voltage - capacitor_voltage) / r1_ohm
    return CircuitResponse(time_s.copy(), source_current, charge)


_SIMULATORS = {
    "series_rc": lambda t, v, params, q0: simulate_series_rc(
        t, v, params["resistance_ohm"], params["capacitance_f"], q0
    ),
    "series_r1_parallel_r2_c": lambda t, v, params, q0: simulate_series_r1_parallel_r2_c(
        t, v, params["r1_ohm"], params["r2_ohm"], params["capacitance_f"], q0
    ),
}


def simulate_circuit(circuit_type, params, time_s, applied_voltage_v, initial_charge_c=0.0):
    """Returns a CircuitResponse, or None if circuit_type has no known
    equation of motion (e.g. "arb_circuit") -- nothing to simulate for those."""
    simulator = _SIMULATORS.get(circuit_type)
    if simulator is None:
        return None
    return simulator(time_s, applied_voltage_v, params, initial_charge_c)


def _analytical_series_rc(t_array, params, q_initial, q_final):
    """Minimum-dissipation charging of an RC circuit is constant current:
    I* = (q_final - q_initial) / t_final, so Q(t) is linear in t and
    V(t) = I*R + Q(t)/C."""
    resistance_ohm, capacitance_f = params["resistance_ohm"], params["capacitance_f"]
    t_final = t_array[-1]
    current = (q_final - q_initial) / t_final
    q_analytical = q_initial + current * t_array
    v_analytical = current * resistance_ohm + q_analytical / capacitance_f
    source_current = np.full_like(t_array, current)
    return v_analytical, q_analytical, source_current


def _analytical_series_r1_parallel_r2_c(t_array, params, q_initial, q_final):
    """Euler-Lagrange solution for R1 in series with (R2 parallel C):
    q_ddot = lambda^2 * q, lambda = sqrt((R1+R2) / (R1 * C^2 * R2^2)),
    with fixed endpoints q(0)=q_initial, q(t_final)=q_final."""
    r1_ohm, r2_ohm, capacitance_f = params["r1_ohm"], params["r2_ohm"], params["capacitance_f"]
    t_final = t_array[-1]
    lam = np.sqrt((r1_ohm + r2_ohm) / (r1_ohm * capacitance_f**2 * r2_ohm**2))
    denom = np.sinh(lam * t_final)

    if denom == 0:
        q_analytical = q_initial + (q_final - q_initial) * (t_array / t_final)
        dq_dt = np.full_like(t_array, (q_final - q_initial) / t_final)
    else:
        q_analytical = (
            q_initial * np.sinh(lam * (t_final - t_array)) / denom
            + q_final * np.sinh(lam * t_array) / denom
        )
        dq_dt = lam * (
            -q_initial * np.cosh(lam * (t_final - t_array)) / denom
            + q_final * np.cosh(lam * t_array) / denom
        )

    v_analytical = r1_ohm * dq_dt + (r1_ohm + r2_ohm) / (capacitance_f * r2_ohm) * q_analytical
    source_current = (v_analytical - q_analytical / capacitance_f) / r1_ohm
    return v_analytical, q_analytical, source_current


_ANALYTICAL_SOLVERS = {
    "series_rc": _analytical_series_rc,
    "series_r1_parallel_r2_c": _analytical_series_r1_parallel_r2_c,
}


def analytical_optimal_protocol(circuit_type, params, t_array, q_initial, q_final):
    """Closed-form minimum-work V(t)/Q(t) plus the work integral VI_integral,
    for circuits with a registered solver. Returns None for circuit_type
    without one (e.g. "arb_circuit") -- the whole point of the BO search is
    to find this without knowing it analytically ahead of time."""
    solver = _ANALYTICAL_SOLVERS.get(circuit_type)
    if solver is None:
        return None
    v_analytical, q_analytical, source_current = solver(t_array, params, q_initial, q_final)
    integrand = v_analytical * source_current
    vi_integral = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t_array)))
    return v_analytical, q_analytical, vi_integral


def _trapz(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _cumulative_trapz(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def capacitor_voltage_from_trace(circuit_type, params, current_a, voltage_v):
    """Pointwise capacitor voltage recovered algebraically from a measured
    (or simulated) trace, when only the total source current I_tot is
    available -- there is no separate capacitor-branch probe. Only defined
    for series_r1_parallel_r2_c: with R1 known, Ohm's law across R1 gives
    the node voltage directly, V_C(t) = V(t) - I_tot(t)*R1, with no need to
    integrate anything (unlike series_rc, whose capacitor voltage is a
    state that has to be built up from I_tot via time integration, and
    arb_circuit, which has no known relation between I_tot and V_C at all).
    Returns None where this recovery isn't defined."""
    if circuit_type != "series_r1_parallel_r2_c":
        return None
    return np.asarray(voltage_v, dtype=float) - np.asarray(current_a, dtype=float) * params["r1_ohm"]


def charge_from_trace(circuit_type, params, time_s, current_a, voltage_v, q_initial=0.0):
    """Recover the final capacitor charge Q_tau from a measured trace that
    only has the total (source) current available. Defaults to
    q_initial + integral(I_tot dt), the total charge delivered by the
    source, which equals the capacitor's charge whenever there is no bypass
    path (series_rc, or an unmodeled arb_circuit -- the best available guess
    without knowing its topology). For series_r1_parallel_r2_c, part of the
    source current bypasses the capacitor through R2, so integral(I_tot dt)
    would overcount the capacitor's charge -- instead this uses
    capacitor_voltage_from_trace at the final sample: Q = C * V_C(t_final)."""
    capacitor_voltage = capacitor_voltage_from_trace(circuit_type, params, current_a, voltage_v)
    if capacitor_voltage is not None:
        return float(params["capacitance_f"] * capacitor_voltage[-1])
    return q_initial + _trapz(current_a, time_s)


def charge_trajectory_from_trace(circuit_type, params, time_s, current_a, voltage_v, q_initial=0.0):
    """Full Q_C(t) trajectory recovered from a measured trace, for plotting.
    For series_r1_parallel_r2_c: pointwise via Ohm's law across R1 (see
    capacitor_voltage_from_trace) -- no integration needed. For series_rc
    (and arb_circuit, best-effort): I_tot is exactly dQ/dt when there's no
    bypass path, so Q_C(t) is q_initial plus the cumulative trapezoidal
    integral of I_tot -- the same rule charge_from_trace uses for its final
    value, just kept at every sample instead of only the last one."""
    capacitor_voltage = capacitor_voltage_from_trace(circuit_type, params, current_a, voltage_v)
    if capacitor_voltage is not None:
        return params["capacitance_f"] * capacitor_voltage
    return q_initial + _cumulative_trapz(current_a, time_s)
