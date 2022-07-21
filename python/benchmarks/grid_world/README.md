Grid World
==========

This directory contains the source code to run the experiment with a grid_world benchmark.

The experiment results are saved under `./logs/`

Usage
-----

### Training

`scripts/run.py` is the script to train a controller for the grid world benchmark. The training log and the trained controller is saved under `./logs/`. See the log message for the exact directory.

The usage example is as follows.

```sh
python ./script/run.py --shield=dynamic # to run an experiment with dynamic shielding

python ./script/run.py --shield=no # to run an experiment without shielding, i.e., the usual RL

python ./script/run.py # to run both with and without shielding
python ./script/run.py --shield=all # the same as the above one
```

### Prediction

`scripts/load.py` is the script to load a trained controller and execute it on the evaluation environment.

The usage example is as follows.

```sh
# Use the model in ./logs/grid_world2-v1_PPO_adaptive_dynamic_shielding-100000-0-1633530955.9108608/best_model/best_model.zip
python ./scripts/load.py ./logs/grid_world2-v1_PPO_adaptive_dynamic_shielding-100000-0-1633530955.9108608/best_model/best_model.zip
```

Benchmark Description
---------------------

GridWorld is a benchmark of a high-level robot control scenario on a discrete grid world environment. This benchmark shows the applicability of dynamic shielding for the environment with a non-ego agent. Moreover, the agent may crash at any position in the arena and the environment is prone to error. 
