"""Live, iteration-by-iteration visualization for the BO experiment."""

import numpy as np
from circuits import simulate_circuit


class ControlPanel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = cfg.enable_control_panel
        self.objectives = []
        self.charges = []
        self.feasible = []
        if not self.enabled:
            return
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        self.ax_voltage, self.ax_current, self.ax_next, self.ax_history = axes.ravel()
        self.ax_charge = self.ax_history.twinx()
        self.figure.canvas.manager.set_window_title("Bayesian Voltage Optimization")
        self.figure.tight_layout(pad=3.0)

    def show_next(self, control_voltages_v, status="Waiting for measurement"):
        if not self.enabled:
            return
        duration = self.cfg.waveform.protocol_duration_s
        control_t = np.linspace(0.0, duration, len(control_voltages_v))
        dense_t = np.linspace(0.0, duration, 1000)
        dense_v = np.interp(dense_t, control_t, control_voltages_v)
        self.ax_next.clear()
        self.ax_next.plot(dense_t, dense_v, color="tab:orange", linewidth=2)
        self.ax_next.scatter(control_t, control_voltages_v, color="tab:red", zorder=3)
        self.ax_next.set(title="Next BO voltage choice", xlabel="Protocol time (s)", ylabel="Voltage (V)")
        self.ax_next.set_xlim(0.0, duration)
        self.ax_next.grid(True, alpha=0.3)
        self.figure.suptitle(status)
        self._refresh()

    def _refresh(self):
        self.figure.tight_layout(pad=3.0)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.05)

    def update(self, measurement, result, next_voltages_v, iteration):
        if not self.enabled:
            return
        self.ax_voltage.clear()
        self.ax_voltage.plot(measurement.time_s, measurement.voltage_v, color="tab:blue")
        self.ax_voltage.set(title="Measured protocol voltage", xlabel="Protocol time (s)", ylabel="Voltage (V)")
        self.ax_voltage.grid(True, alpha=0.3)

        self.ax_current.clear()
        self.ax_current.plot(measurement.time_s, measurement.current_a,
                             color="tab:green", label="Measured current")
        if self.cfg.circuit.show_theoretical_current:
            theory = simulate_circuit(
                self.cfg.circuit.circuit_type,
                self.cfg.circuit.params,
                measurement.time_s,
                measurement.voltage_v,
                self.cfg.circuit.initial_charge_c,
            )
            if theory is not None:
                self.ax_current.plot(theory.time_s, theory.current_a, "--",
                                     color="tab:orange", linewidth=1.8,
                                     label="Theoretical circuit-model current")
        self.ax_current.set(title="Measured protocol current", xlabel="Protocol time (s)", ylabel="Current (A)")
        self.ax_current.grid(True, alpha=0.3)
        self.ax_current.legend(loc="best")

        self.objectives.append(float(result["VI_integral"]))
        self.charges.append(float(result["Q_tau"]))
        self.feasible.append(bool(result["feasible"]))
        self.ax_history.clear()
        self.ax_charge.clear()
        self.ax_history.yaxis.set_label_position("left")
        self.ax_history.yaxis.tick_left()
        self.ax_history.spines["left"].set_position(("axes", 0.0))
        self.ax_charge.yaxis.set_label_position("right")
        self.ax_charge.yaxis.tick_right()
        self.ax_charge.spines["right"].set_position(("axes", 1.0))
        indices = np.arange(len(self.objectives))
        colors = ["tab:green" if value else "tab:red" for value in self.feasible]
        self.ax_history.plot(indices, self.objectives, color="0.5", linewidth=1)
        self.ax_history.scatter(indices, self.objectives, c=colors, s=24)
        self.ax_history.set(title="Work history (green = charge feasible)", xlabel="Evaluation", ylabel="Integral V I dt (J)")
        self.ax_history.tick_params(axis="y", colors="0.25")
        self.ax_history.grid(True, alpha=0.3)
        self.ax_charge.plot(indices, self.charges, color="tab:purple", marker="o",
                            markersize=3, label="Measured Q_tau")
        self.ax_charge.axhline(float(result["Q_target"]), color="tab:red",
                               linestyle="--", label="Target Q_tau")
        self.ax_charge.set_ylabel("Q_tau (C)", color="tab:purple")
        self.ax_charge.tick_params(axis="y", colors="tab:purple")
        self.ax_charge.legend(loc="upper right")

        status = (f"Evaluation {iteration} | {result['phase']} | "
                  f"Q={result['Q_tau']:.4g} C | target={result['Q_target']:.4g} C | "
                  f"feasible={result['feasible']}")
        self.show_next(next_voltages_v, status)

    def finish(self):
        if not self.enabled:
            return
        self.figure.suptitle("Experiment complete - close this window to exit")
        self._refresh()
        self.plt.ioff()
        self.plt.show()
