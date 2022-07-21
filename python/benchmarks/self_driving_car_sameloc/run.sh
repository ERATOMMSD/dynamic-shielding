#!/bin/bash
# no shield
python -W ignore run_self_driving_car_dynamic_shield.py 0
sleep 2
# padding
python -W ignore run_self_driving_car_dynamic_shield.py 1
sleep 2
# dynamic shield
# need to launch java learnlib ../../../../java/learnlib-py4j-example-1.0-SNAPSHOT.jar
python -W ignore run_self_driving_car_dynamic_shield.py 2
sleep 2