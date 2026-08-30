# Bayesian voltage optimization

The code is separated into pure optimization/waveform logic and VISA instrument I/O:

- `bo.py`: Xopt constrained Expected Improvement following the reference notebook.
- `waveform.py`: interpolates the protocol and appends its ramp-down and zero hold.
- `instruments.py`: Agilent ARB upload and Infiniium acquisition.
- `measurement.py`: applies one proposed voltage and returns a calibrated trace with `t=0` at the start of the BO protocol.
- `storage.py`: saves every trace and the compact optimization history.
- `config.py`: all user-editable settings.
- `run_experiment.py`: the sequential experiment loop.

Every physical waveform is:

1. the interpolated BO voltage protocol, starting at the ARB period boundary,
2. a gradual ramp from its final voltage to `0 V`,
3. an adjustable slow negative neutralization pulse and return to `0 V`,
4. an adjustable final `0 V` hold.

ARB sync output is measured on oscilloscope channel 3. Its positive 0.5 V crossing defines protocol `t=0`. The ramp-down, negative neutralization pulse, and zero hold are excluded from the BO objective.

The optimizer follows `BO_discretize_points_R1_R2_parallel_C.ipynb`: first measure a linear 10-point ramp, use its transferred charge as the lower charge bound, measure 30 random protocols, then use Xopt constrained Expected Improvement. It minimizes `integral(V*I dt)` subject to measured `Q_tau = integral(I dt)` remaining at or above the linear-ramp value.

Install `numpy`, `pandas`, `xopt`, and `pyvisa` plus the appropriate VISA backend, verify the two VISA addresses and gains in `config.py`, then run:

```powershell
python run_experiment.py
```

Before connecting a device, validate voltage limits, channel scaling, current-amplifier gain, instrument load setting, and trigger behavior on an oscilloscope.

Create an evolution GIF from the latest timestamped run in the configured circuit folder:

```powershell
python make_evolution_video.py
```

Or select a specific run with `python make_evolution_video.py --history <path-to-bo_history.csv>`.

To select a folder and automatically use its newest history file, run
`python make_evolution_video.py --data-folder <folder-containing-the-CSVs>`.
