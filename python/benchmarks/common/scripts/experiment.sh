#!/bin/bash


benchmark=$1
steps=$2
repetitions=$3

sbatch --array=1-$repetitions launch.sh $benchmark $steps no 
sbatch --array=1-$repetitions launch.sh $benchmark $steps pre-adaptive
sbatch --array=1-$repetitions launch.sh $benchmark $steps safe-padding 
