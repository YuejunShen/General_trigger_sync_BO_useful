"""Sequential BO measurement loop. Edit config.py, then run this file."""

from bo import BayesianVoltageOptimizer
from config import ExperimentConfig
from control_panel import ControlPanel
from make_evolution_video import default_output_path, make_video
from measurement import apply_voltage_and_measure
from storage import save_measurement, save_run_config


def main() -> None:
    cfg = ExperimentConfig()
    save_run_config(cfg)
    optimizer = BayesianVoltageOptimizer(cfg.waveform, cfg.bo, cfg.circuit)
    panel = ControlPanel(cfg)
    proposed_voltage = optimizer.propose()
    panel.show_next(proposed_voltage, "Initial linear protocol")
    for iteration in range(cfg.bo.total_measurements):
        print(f"Iteration {iteration}: applying {proposed_voltage}")
        measurement = apply_voltage_and_measure(proposed_voltage, cfg)
        result = optimizer.observe(
            measurement.applied_control_voltages_v,
            measurement.time_s,
            measurement.current_a,
            measurement.voltage_v,
        )
        save_measurement(measurement, iteration, result, cfg)
        print(f"VI={result['VI_integral']:.8g} J, Q_tau={result['Q_tau']:.8g} C, "
              f"feasible={result['feasible']}, phase={result['phase']}")
        proposed_voltage = optimizer.propose()
        panel.update(measurement, result, proposed_voltage, iteration)

    try:
        video_path = make_video(cfg.history_file, default_output_path(cfg.history_file))
        print(f"Evolution video saved to {video_path}")
    except Exception as exc:
        print(f"Could not generate evolution video: {exc}")

    panel.finish()


if __name__ == "__main__":
    main()
