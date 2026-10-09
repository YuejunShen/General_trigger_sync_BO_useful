# Bayesian voltage optimization

This project runs a sequential, hardware-in-the-loop Bayesian optimization (BO) experiment. It programs an arbitrary waveform generator (ARB), acquires current, voltage, and sync traces from an oscilloscope, and minimizes measured electrical work while enforcing a final-charge constraint.

## Experiment sequence

Each run performs:

1. one linear reference protocol from `linear_minimum_voltage_v` to `linear_maximum_voltage_v`;
2. `initial_random_measurements` random protocols;
3. `bo_steps` constrained Expected Improvement proposals from Xopt.

The first measurement defines `Q_target`. BO minimizes `VI_integral = integral(V I dt)` while searching for protocols whose final capacitor charge is at least the configured lower bound. `q_lower_relative_tolerance` adjusts the BO search boundary; the saved `feasible` value always means `Q_tau >= Q_target`.

The uploaded ARB waveform contains the interpolated BO protocol, a ramp to zero, a negative neutralization pulse, a return to zero, and a final zero hold. Only the BO protocol interval contributes to the objective and charge calculation. The ARB sync output is acquired on the configured trigger channel; its positive threshold crossing defines protocol `t = 0`.

## Hardware and software setup

The instrument driver currently targets an Agilent-compatible ARB and Infiniium oscilloscope through VISA. The default scope channels are:

- channel 1: current-amplifier output;
- channel 2: device voltage;
- channel 3: ARB sync/trigger output.

Install the Python dependencies and a compatible VISA backend such as NI-VISA:

```powershell
python -m pip install numpy pandas packaging "xopt>=3.2.1" pyvisa matplotlib pillow
```

Pillow supports GIF output. MP4 output also requires FFmpeg on `PATH`.

`run_in_xopt_env.ps1` is a convenience launcher for this lab computer. It uses the fixed interpreter `C:\Users\16502\.conda\envs\bto-xopt-latest\python.exe`. Edit `$python` for another machine or environment, or use the active environment directly.

## Configure a run

Edit `config.py` before connecting or energizing a device.

### `InstrumentConfig`

Set both VISA addresses, channel assignments, oscilloscope scales and offsets,
trigger threshold, and voltage scaling. Set `use_current_amplifier = True` when
channel 1 receives a current-amplifier output; current is then calculated as
`V_scope * current_termination_correction * current_gain_a_per_v`. Set it to
`False` when the circuit is connected directly to the scope's 50-ohm input;
current is then calculated as `V_scope / current_input_impedance_ohm`. The
current driver currently requires `current_input_impedance_ohm = 50.0`.

### `WaveformConfig`

Check the protocol duration and control-point count, ARB sample rate, BO voltage bounds, reference-ramp bounds, ramp-down timing, neutralization pulse, and final zero hold.

### `BOConfig`

Set the random-measurement and BO-step counts, charge-bound tolerance, random seed behavior, and numerical optimizer limits. A run makes `1 + initial_random_measurements + bo_steps` measurements (131 with the current defaults).

### `CircuitConfig`

Choose a `circuit_type` and provide its matching parameters:

- `series_rc`: `resistance_ohm` and `capacitance_f` (uncomment/add `resistance_ohm` in the current config);
- `series_r1_parallel_r2_c`: `r1_ohm`, `r2_ohm`, and `capacitance_f`;
- `arb_circuit`: no model parameters; analytical and simulated-current comparisons are skipped.

The circuit selection affects how final capacitor charge is inferred from the measured trace, so it must match the physical setup.

### `ExperimentConfig`

By default, `data_mother_path` is created automatically as
`D:\Ferroelectric_BTO_optimization\YYYYMMDD` using the run's start date. Override
it in `config.py` only when a different output root is needed. Also review
`scope_time_step_s`, `scope_periods`, `output_on_wait_s`, and
`enable_control_panel`.

Each run creates `BO_<circuit-parameters>_<HHMM>/` beneath `data_mother_path`. It contains a timestamped configuration JSON, one trace CSV per evaluation, a compact BO history CSV, and, when rendering succeeds, an MP4 evolution video.

## Run the experiment

Before starting, verify voltage limits, generator load behavior, channel scaling and offsets, current-amplifier gain, 50-ohm termination, wiring, and the sync-trigger edge on an oscilloscope.

Run with the active environment:

```powershell
python run_experiment.py
```

To sweep the number of protocol discretization points, edit
`DISCRETIZATION_POINTS` and `REPEATS` near the top of
`run_parameter_sweep.py`, then run:

```powershell
python run_parameter_sweep.py
```

Each setting uses an independent optimizer and output folder. The live panel's
**Stop safely** button finishes the active measurement, closes the instrument
sessions normally, and then ends either the single run or the entire sweep.

Or, after adapting its interpreter path:

```powershell
.\run_in_xopt_env.ps1
```

The generator output is enabled only during acquisition and disabled during cleanup. At completion, the program attempts to create an MP4 and leaves the live control-panel window open until it is closed. Set `enable_control_panel = False` for headless operation.

## Create an evolution video

Use an exact history file:

```powershell
python make_evolution_video.py --history <path-to-bo_history.csv>
```

Or select the newest history in a run folder:

```powershell
python make_evolution_video.py --data-folder <folder-containing-run-CSVs>
```

Optional arguments are `--output <file.mp4|file.gif>` and `--fps <frames-per-second>`. With neither selection argument, the script uses the user-editable `DATA_FOLDER` near the top of `make_evolution_video.py`.

## Code layout

- `config.py` — experiment, hardware, waveform, circuit, and BO settings.
- `run_experiment.py` — sequential experiment loop.
- `waveform.py` — protocol interpolation and waveform-tail construction.
- `instruments.py` — VISA communication with the ARB and oscilloscope.
- `measurement.py` — acquisition, calibration, and protocol alignment.
- `bo.py` — constrained Expected Improvement and trace metrics.
- `circuits.py` — circuit-specific charge inference, simulation, and analytical references.
- `storage.py` — configuration, trace, and history persistence.
- `control_panel.py` — live plots during a run.
- `make_evolution_video.py` — MP4/GIF rendering from saved data.
