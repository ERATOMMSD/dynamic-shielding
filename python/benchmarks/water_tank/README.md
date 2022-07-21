How to reproduce the experiments
--------------------------------

```sh
python ./train.py --steps 500000  --shield=pre-adaptive # to run an experiment with dynamic shielding
python ./train.py --steps 500000  --shield=safe-padding # to run an experiment with safe padding
python ./train.py --steps 500000  --shield=no # to run an experiment without shielding
```
