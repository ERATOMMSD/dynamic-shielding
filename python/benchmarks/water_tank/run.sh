#!/bin/bash

entry=water_tank_train.py
depths=(0 1 3 5 7)
penalties=(1 10 100 1000)
steps=$1
repetitions=$2

echo "Running experiment of $steps steps ($repetitions repetitions)..."

sleep 10

for t in {1..$repetitions}; do
  echo "Starting iteration $t..."
  for i in ${!penalties[@]}; do
    echo "Starting no shield with penalty ${penalties[$i]}..."
    sleep 1
    python $entry --shield no --penalties ${penalties[$i]} --steps $steps
  done
  for i in ${!depths[@]}; do
    echo "Starting dynamic shield with depth ${depths[$i]}..."
    sleep 1
    python $entry --shield dynamic --depths ${depths[$i]} --steps $steps
  done
done