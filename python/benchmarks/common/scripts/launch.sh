#!/bin/bash

#SBATCH --partition=aloha --cpus-per-task=8 --gpus=1

benchmark=$1
steps=$2
shield=$3

activate(){
 . ~/safe-rl-shielding-copy/venv/bin/activate
}

activate

cd ~/safe-rl-shielding-copy/python/benchmarks/${benchmark}/

python3 train.py --shield $shield --steps $steps
