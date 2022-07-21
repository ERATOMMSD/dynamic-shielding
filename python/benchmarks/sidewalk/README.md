Sidewalk
========

This directory contains the sidewalk benchmark and some scripts for its evaluation.

Usage
-----

### Training

`train.py` is the script to train a controller for the sidewalk benchmark. The training log and the trained controller is saved under `./logs/`. See the log message for the exact directory.

The usage example is as follows.

```sh
# Train a controller with dynamic shielding + adaptive min_depth
python ./train.py --shield=adaptive

# Train a controller with dynamic shielding + static min_depth. The min_depth is defined in helper.py.
python ./train.py --shield=dynamic

# Train a controller without shielding, i.e., the usual RL
python ./train.py --shield=no

# Train a controller with safe padding in [Hasanbeig+, AAMAS'20]
python ./train.py --shield=safe_padding
```

### Prediction

`load.py` is the script to load a trained controller and execute it on the evaluation environment.

The usage example is as follows.

```sh
# Use the model in ./logs/MacBook-Pro-3.local/ranged_deterministic_sidewalk-v2/Oct04_19:19:00/ranged_deterministic_sidewalk-v2_dynamic_shield-depth1/best_model/best_model.zip
python ./load.py ./logs/MacBook-Pro-3.local/ranged_deterministic_sidewalk-v2/Oct04_19:19:00/ranged_deterministic_sidewalk-v2_dynamic_shield-depth1/best_model/best_model.zip
```

Benchmark Description
---------------------

Sidewalk is a benchmark to show the applicability of dynamic shielding for the environment with randomness. Since the dynamic shield assumes that the training environment is deterministic, we use deterministic environment called `ranged_deterministic_sidewalk` in the training and use it in the non-deterministic environment called `sidewalk`. See the following for the detail of these benchmarks.

- `ranged_deterministic_sidewalk`: Basically the same as the standard Sidewalk benchmark but the seed of the random number generator is fixed at each reset. The seed is sampled from the specified range. Moreover, the chosen seed is observed as the player 2 action of the initial step.
- `sidewalk`: The standard sidewalk benchmark with domain randomization

