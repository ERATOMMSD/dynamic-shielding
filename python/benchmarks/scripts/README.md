Scripts for paper writing
===========================

This directory contains scripts to summarize the experiment results toward paper writing.

NOTE: This packages required for the scripts in directory is different from that of the other part of this directory. Please set up another environment following the following instructions.

Setup
------

### Requirements

### Virtual environment
In this repository, there are two "requirements" files. 
- `requirements.txt`: original requirements of the *safe-rl-shielding* benchmarks in `./envs/` + *pyeda*
- `requirements-sb.txt`: requirements for stable-baselines benchmarks in `./python/benchmarks/` 

All benchmarks inside `./python/benchmarks/` use  `requirements-sb.txt` except for self_driving_car, which requires some libraries that are used in the original benchmarks.

As for the original benchmarks, since the required OpenCV is too old, the latest python does not work. We can set up the suitable python environment using pyenv-virtualenv and install the required packages as follows.

```sh
pyenv virtualenv 3.6.10 safe-rl-shielding
pyenv local safe-rl-shielding
pyenv exec pip install -r requirements.txt
```

As for the stable-baselines benchmarks, Python 3.6.8+ is required. 

```sh
pyenv virtualenv 3.6.8 shielded-automata-learning
pyenv local shielded-autamata-learning
pyenv exec pip install -r requirements.txt
```
