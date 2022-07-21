Car racing
==========

This directory contains the car racing benchmark and some scripts for its evaluation.

Usage
-----

### Training

`scripts/run.py` is the script to train a controller for the sidewalk benchmark. The training log and the trained controller is saved under `./logs/`. See the log message for the exact directory.

The usage example is as follows.

```sh
# Train a controller with dynamic shielding + adaptive min_depth
python ./scrips/run.py --shield adaptive

# Train a controller with dynamic shielding + static min_depth. The min_depth is defined in helper.py.
python ./scrips/run.py --shield dynamic

# Train a controller without shielding, i.e., the usual RL
python ./scrips/run.py --shield=no

# Train a controller with safe padding in [Hasanbeig+, AAMAS'20]
python ./scrips/run.py --shield=safe_padding
```

### Prediction

`scripts/load.py` is the script to load a trained controller and execute it on the evaluation environment.

The usage example is as follows.

```sh
# Use the model in ./logs/aloha01.group-mmm.org/discrete_car_racing-v4/Oct14_11:04:25/discrete_car_racing-v4_no_shield/best_model/best_model.zip
python ./load.py ./logs/aloha01.group-mmm.org/discrete_car_racing-v4/Oct14_11:04:25/discrete_car_racing-v4_no_shield/best_model/best_model.zip
```

Benchmark Description
---------------------

Car racing is a benchmark to show the applicability of dynamic shielding for the environment with complex physical dynamics and the application of the conventional shielding approach is hard. The physical dynamics of this benchmark is relatively complex, e.g., considering different road friction in the grass area and the road area, and simulated with box2d, a physical engine for 2d environment.

Note
----

Since car_racing benchmark requires image rendering, we need to use `xvfb-run` if we run it on a headless server. The following shows an example.

```sh
xvfb-run -s "-screen 0 1400x900x24" pyenv exec python ./benchmarks/car_racing/scripts/run.py
```
