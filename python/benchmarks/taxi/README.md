Taxi
=====

This directory contains the Taxi benchmark taken from OpenAI's [`gym`](https://gym.openai.com/). We modified the
original benchmark so that the arena is fixed.

How to reproduce the experiments
--------------------------------

```sh
python ./train.py --steps 200000  --shield=pre-adaptive # to run an experiment with dynamic shielding
python ./train.py --steps 200000  --shield=safe-padding # to run an experiment with safe padding
python ./train.py --steps 200000  --shield=no # to run an experiment without shielding
```
