Self Driving Car
================

This directory contains a variant of the [self-driving-car](https://github.com/safe-rl/safe-rl-shielding/tree/master/envs/self-driving-car) benchmark.
We remark that this requires an environment with the packages in `../../../requirements.txt`. 

Usage
-----

```sh
# Make directories to contain logs
mkdir -p ./logs/{NO_SHIELD,SHIELD,PADDING}/{TRAIN,TEST}
# Run all the experiment setting
./run.sh
```

See `const.py` for the hyperparameters.
