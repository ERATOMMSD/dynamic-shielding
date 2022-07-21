import sys
import os.path as osp
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT+"envs/self_driving_car/")


from python.benchmarks.self_driving_car_sameloc.const import *
from python.benchmarks.self_driving_car_sameloc.env_road_new_cw import Env
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from envs.self_driving_car.rl.memory import SequentialMemory
from python.benchmarks.self_driving_car_sameloc.dynamic_dqn_abstraction import DQNAgent
from python.benchmarks.self_driving_car_sameloc.self_driving_car_dynamic_shield_abstraction import SelfDrivingCarDynamicShieldAbs
from python.benchmarks.self_driving_car_sameloc.self_driving_car_safe_padding import SelfDrivingCarSafePadding
from benchmarks.common.generic import launch_gateway
from py4j.java_gateway import JavaGateway, CallbackServerParameters
from datetime import datetime


nb_actions = 3  # number of actions (go straight, left, right)
observation_space = (4,)
model = Sequential()
model.add(Flatten(input_shape=(1,) + observation_space))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
memory = SequentialMemory(limit=500000, window_length=1)

gateway = launch_gateway()

start_time = datetime.now()
REUSE_SHIELD = False
dy_shield = None
learning_rate = 3e-4

SHIELD_MODE = int(sys.argv[1])
# print('SHIELD_MODE:', MODE_NAME_LIST[SHIELD_MODE])

if SHIELD_MODE == NO_SHIELD:
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=learning_rate, shield=None, dynamic_shield=None)
    ENV_NAME = 'no_shield'
    env = Env(env_label=ENV_NAME, big_neg=False)
elif SHIELD_MODE == DYNAMIC_PREEMPTIVE_MODE:
    ltl_formula = 'G(s)'
    dy_shield = SelfDrivingCarDynamicShieldAbs(ltl_formula, gateway)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=learning_rate, shield=None, dynamic_shield=dy_shield)
    ENV_NAME = 'dynamic_preemptive_shield'
    env = Env(env_label=ENV_NAME, big_neg=False)
elif SHIELD_MODE == SAFE_PADDING:
    ltl_formula = 'G(s)'
    pd_shield = SelfDrivingCarSafePadding(ltl_formula)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=learning_rate, shield=None, dynamic_shield=pd_shield)
    ENV_NAME = 'safe_padding'
    env = Env(env_label=ENV_NAME, big_neg=False)
else:
    raise NotImplementedError('shield mode ' + str(SHIELD_MODE) + ' not implemented')

dqn.compile(Adam(lr=learning_rate), metrics=['mae'])
score = 0
counter = 0
start = datetime.now()
score_log = []

dqn.fit_test(env, visualize=False, verbose=0, SHIELD_MODE=SHIELD_MODE)
# print('time = ',  datetime.now() - start_time)

gateway.close()

