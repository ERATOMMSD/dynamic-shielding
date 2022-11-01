Dynamic Shielding
=================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python Unittest](https://github.com/ERATOMMSD/dynamic-shielding/actions/workflows/unittest.yml/badge.svg?branch=main)](https://github.com/ERATOMMSD/dynamic-shielding/actions/workflows/unittest.yml)

This is the source code repository for an implementation of dynamic shielding for reinforcement learning.

Setup
------

In order to ensure reproducibility of the results, we suggest using two different virtual environments.
One for the benchmark taken from [safe-rl-shielding](https://github.com/safe-rl/safe-rl-shielding/) and the other for the other benchmarks.

### Virtual environment

In this repository, there are two "requirements" files. 
- `requirements.txt`: original requirements of the *safe-rl-shielding* benchmarks in `./envs/` + *pyeda*
- `requirements-sb.txt`: requirements for stable-baselines benchmarks in `./python/benchmarks/` 

All benchmarks inside `./python/benchmarks/` use  `requirements-sb.txt` except for self_driving_car, which requires some libraries that are used in the original benchmarks.

Since some of the benchmarks requires old OpenCV, the latest python does not work. We can set up the suitable python environment using pyenv-virtualenv and install the required packages as follows.

```sh
pyenv virtualenv 3.6.10 safe-rl-shielding
pyenv local safe-rl-shielding
pyenv exec pip install -r requirements.txt
```

As for the stable-baselines benchmarks, Python 3.6.8+ is required. 

```sh
pyenv virtualenv 3.6.8 shielded-automata-learning
pyenv local shielded-autamata-learning
pyenv exec pip install -r requirements-sb.txt
```

We note that the use of pyenv itself is optional. You can use `venv` to make separated environments.

Usage
-----

1. Compile the java part of our code with `cd java && mvn package`
2. Set up the environment, for example, with `python3.6 -v venv .venv`
3. Activate the environment, for example, with `. .venv/bin/activate`
4. Update `pip` with `pip install --upgrade pip`
5. Install the dependencies with `pip install -r requirements-sb.txt`
   - You may need some extra care if you want to use GPUs
6. Run a script to train a controller, for example, `cd python/benchmarks/grid_world/ && python ./scripts/run.py`
   - You can disable GPUs. See [this stack overflow post](https://stackoverflow.com/questions/53266350/how-to-tell-pytorch-to-not-use-the-gpu) for the detail.
   - We remark that you should repeat the experiment sufficiently many times due to its randomness.
7. You can see the result using tensorboard


Note
----

### How to install Spot for pyenv virtualenv or venv

By default, the python library for Spot is installed under `/usr/local/`. In order to use Spot in pyenv virtualenv, the python library must be installed under `$HOME/.pyenv/versions/shielded-learning/`. By the following modification of `configure` in spot, we can install the python library in the appropriate directory.

```sh
sed -i 's:PYTHON_PREFIX=.*:PYTHON_PREFIX="$HOME/.pyenv/versions/shielded-learning/":;s:PYTHON_EXEC_PREFIX=.*:PYTHON_EXEC_PREFIX="$HOME/.pyenv/versions/shielded-learning/":;' configure  && ./configure --prefix ~/.pyenv/versions/shielded-learning/ 
```

If you use venv, the command should be as follows.

``` sh
sed -i 's:PYTHON_PREFIX=.*:PYTHON_PREFIX="$HOME/dynamic-shielding/venv/":;s:PYTHON_EXEC_PREFIX=.*:PYTHON_EXEC_PREFIX="$HOME/dynamic-shielding/venv/":;' configure && ./configure --prefix ~/dynamic-shielding/venv/
```

Alternatively, spot location can be added to the virtual environment by adding the following line inside `./venv/lib/python3.6/site-packages/distutils-precedence.pth` 

```sh 
import sys; sys.path.append('/usr/local/lib/python3.6/site-packages/');
```

Contributors (to the source code)
---------------------------------

- [Masaki Waga](https://maswag.github.io/) [@MasWag](https://github.com/MasWag)
- [Ezequiel Castellano](https://www.linkedin.com/in/ezequiel-castellano-7076962b/?originalSubdomain=jp) [@ezecastellano](https://github.com/ezecastellano)
- [Sasinee Pruekprasert](https://psasinee.github.io/) [@psasinee](https://github.com/psasinee)
- [Stefan Klikovits](https://klikovits.net/) [@stklik](https://github.com/stklik)

Citation
--------

If you want to cite our paper, please use the following .bib file.

```
@inproceedings{WCPKTH22,
  author    = {Masaki Waga and
               Ezequiel Castellano and
               Sasinee Pruekprasert and
               Stefan Klikovits and
               Toru Takisaka and
               Ichiro Hasuo},
  editor    = {Ahmed Bouajjani and
               Luk{\'{a}}s Hol{\'{\i}}k and
               Zhilin Wu},
  title     = {Dynamic Shielding for Reinforcement Learning in Black-Box Environments},
  booktitle = {Automated Technology for Verification and Analysis - 20th International
               Symposium, {ATVA} 2022, Virtual Event, October 25-28, 2022, Proceedings},
  series    = {Lecture Notes in Computer Science},
  volume    = {13505},
  pages     = {25--41},
  publisher = {Springer},
  year      = {2022},
  url       = {https://doi.org/10.1007/978-3-031-19992-9\_2},
  doi       = {10.1007/978-3-031-19992-9\_2}
}
```

Acknowledgments
---------------

The source code under `envs` are originally from https://github.com/safe-rl/safe-rl-shielding/, which is distributed under MIT license. Some of the source code under `java/` are 
originally from https://github.com/mtf90/learnlib-py4j-example, which is distributed under Apache-2.0 license.

Reference
---------

* Dynamic Shielding for Reinforcement Learning in Black-Box Environments. Masaki Waga, Ezequiel Castellano, Sasinee Pruekprasert, Stefan Klikovits, Toru Takisaka, and Ichiro Hasuo
