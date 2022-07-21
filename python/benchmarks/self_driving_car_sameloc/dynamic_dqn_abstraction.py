import os
import sys
sys.path.append("../../../../envs/self-driving-car/")
import keras.backend as K
from keras.layers import Lambda, Input, Layer, Dense
from policy_new import BoltzmannQPolicy, GreedyQPolicy
from rl.util import *
from rl.keras_future import Model
import warnings
from copy import deepcopy
import numpy as np
from keras.callbacks import History
from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
import math
from const import *
from datetime import datetime


def get_input(obs, step):
    if not USE_GRID:
        return 0, ''
    else:
        x = obs[0] * XSIZE
        y = obs[1] * YSIZE
        #  obs[3] is cos(theta), obs[2] is sin(theta), theta = 0 means heading down
        dr = math.acos(obs[3])  # / (0.25 * math.pi)
        if obs[2] < 0:  # sin < 0
            dr = -dr
        if dr < 0:
            dr += math.pi * 2
        grid = GRID
        x_abs, y_abs = math.floor(x/grid), math.floor(y/grid)
        dr -= math.pi/8
        dr_abs = round(dr/ (math.pi/4), 0)
        inp = int(x_abs * (math.ceil(YSIZE / GRID)+1) * 9 + y_abs *9 + dr_abs)
        info = str(step) + ': xy='+ str(x_abs) +',' +str(y_abs)+' deg='+str(dr_abs)+'('+str(int( math.degrees(math.acos(obs[3])) ))+')'
        return inp, info


def get_current_area(obs):
    XSIZE = 480.
    YSIZE = 480.
    areas = [AREA_1, AREA_2, AREA_3, AREA_4]
    x = obs[0] * XSIZE
    y = obs[1] * YSIZE
    for area in areas:
        if area[0] <= x < area[1] and area[2] <= y < area[3]:
            return areas.index(area)
    return None


class Agent(object):
    def __init__(self, processor=None, shield=None, maze=False, manual=False, dynamic_shield=None):
        self.processor = processor
        self.training = False
        self.step = 0
        self.shield = shield
        self.maze = maze
        self.manual = manual
        self.dynamic_shield = dynamic_shield

    def get_config(self):
        return {}

    def fit_test(self, env, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, log_interval=10000, stepper=False, SHIELD_MODE=0):
        FILE_NAME = './logs/' + str(MODE_NAME_LIST[SHIELD_MODE]) +'/'
        TRAIN_RESULT_FILE_NAME = FILE_NAME + 'TRAIN/'
        TEST_RESULT_FILE_NAME = FILE_NAME + 'TEST/'

        count_shield = 0
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))
        self.stepper = stepper
        callbacks = [] if not callbacks else callbacks[:]
        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': TOTAL_TRAINING_STEP,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        callbacks.on_train_begin()

        episode = 1
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        old_accidents = env.accidents
        EPISODE_TYPE_TRAIN = 0
        EPISODE_TYPE_TEST = 1
        current_episode_type = EPISODE_TYPE_TRAIN
        next_episode_test = False
        self.training = True
        current_area = None
        corner = None
        mealy_size = 0
        game_size = 0
        shield_min_depth = 0
        game_safe_states = 0
        try:
            i = 0
            while os.path.exists(TRAIN_RESULT_FILE_NAME+str(i)+'.csv'):
                i += 1
            # print('file =', i)
            local_TRAIN_RESULT_FILE_NAME = TRAIN_RESULT_FILE_NAME + str(i)+'.csv'
            local_TEST_RESULT_FILE_NAME = TEST_RESULT_FILE_NAME + str(i) + '.csv'

            target_train = open(local_TRAIN_RESULT_FILE_NAME , 'w')
            target_train.write(local_TRAIN_RESULT_FILE_NAME)
            target_train.write("\n")
            target_train.write(
                'episode, score, #steps, #all_steps, #of_accidents, #of_shielded, episode_duration, total_duration, corner, mealy_size, game_size, min_depth, game_safe_states')
            target_train.write("\n")

            target_test = open(local_TEST_RESULT_FILE_NAME, 'w')
            target_test.write(local_TEST_RESULT_FILE_NAME)
            target_test.write("\n")
            target_test.write(
                'episode, score, #steps, #all_steps, #of_accidents, #of_shielded, episode_duration, total_duration, corner, mealy_size, game_size, min_depth, game_safe_states')
            target_test.write("\n")

            action = 0
            count_no_acc = 0
            initial_time = datetime.now()
            while self.step <= TOTAL_TRAINING_STEP:
                penalty = 0
                if observation is None:  # start of a new episode
                    start_time = datetime.now()  # start time of this run
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.
                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    env_reset = env.reset()
                    loc = env_reset[1]
                    observation = deepcopy(env_reset[0])
                    if current_episode_type == EPISODE_TYPE_TRAIN:
                        dynamic_shield = self.dynamic_shield
                    else:
                        dynamic_shield = None
                    #reset the environment and the shields
                    if (SHIELD_MODE == DYNAMIC_PREEMPTIVE_MODE or SHIELD_MODE == SAFE_PADDING)\
                            and current_episode_type == EPISODE_TYPE_TRAIN:
                        if SHIELD_MODE == SAFE_PADDING:
                            dynamic_shield.set_initial_observation(observation)
                        else:
                            mealy_size = len(dynamic_shield.mealy.getStates())
                            game_size = len(dynamic_shield.safety_game.getStates())
                            shield_min_depth = dynamic_shield.learner.min_depth
                            game_safe_states = len(dynamic_shield.safety_game.safeStates )
                        dynamic_shield.reset()
                        new_inp, inp_info = get_input(observation, episode_step)
                        assert loc == 0
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None
                    old_area = current_area
                    current_area = get_current_area(observation)
                    if current_area is None:
                        current_area = old_area
                    corner = 0
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                callbacks.on_step_begin(episode_step)
                use_shield = False
                if self.manual:
                    raise NotImplementedError('manual dynamic shield not implemented')
                if SHIELD_MODE == DYNAMIC_PREEMPTIVE_MODE or SHIELD_MODE == SAFE_PADDING:
                    if self.maze:
                        raise NotImplementedError('maze shield not implemented')
                    if current_episode_type == EPISODE_TYPE_TEST:  # test without dynamic shield
                        banned_actions = []
                    else:
                        assert dynamic_shield is not None
                        banned_actions = [aa for aa in range(self.nb_actions) if aa not in dynamic_shield.preemptive()]
                        if len(banned_actions) > 0:
                            count_shield += 1
                            use_shield = True
                    action = self.forward(observation, banned_actions=banned_actions)
                elif SHIELD_MODE == NO_SHIELD:
                    action = self.forward(observation)
                else:
                    raise NotImplementedError('shield mode ' + SHIELD_MODE + ' not implemented')
                reward = 0.
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action, get_input, episode_step, use_shield, episode, current_episode_type)
                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r + penalty, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r + penalty
                    if done:
                        break

                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }

                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                if self.training:
                    self.step += 1
                    if self.step % TEST_EVERY_STEP == 0:
                        next_episode_test = True

                new_inp, inp_info = get_input(observation, episode_step)
                if SHIELD_MODE != NO_SHIELD and \
                        current_episode_type == EPISODE_TYPE_TRAIN:
                    if SHIELD_MODE == SAFE_PADDING:
                        if old_accidents < env.accidents:
                            count_no_acc += 1
                            dynamic_shield.move(action, 0, new_inp + 10000000, observation)
                        else:
                            dynamic_shield.move(action, 0, new_inp, observation)
                    else:
                        if old_accidents < env.accidents:  # collision
                            count_no_acc += 1
                            dynamic_shield.move(action, 0, new_inp + 10000000)
                        else:
                            dynamic_shield.move(action, 0, new_inp)

                old_accidents = env.accidents
                assert current_area is not None
                new_area = get_current_area(observation)
                if new_area == (current_area+1) % 4:
                    corner += 1
                    current_area = new_area
                elif new_area == (current_area-1) % 4:  # runing clock-wise
                    corner = 0
                    current_area = new_area
                elif new_area != current_area and new_area is not None:
                    raise ValueError()
                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    # all self.backward to update the NN. if self.training == False, the NN will not be update
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)
                    current_time = datetime.now()
                    if current_episode_type == EPISODE_TYPE_TRAIN:
                        assert self.training
                        target_train.write(
                            str(episode) + ',' + str(episode_reward) + ',' + str(episode_step) + ','
                            + str(self.step) + ',' + str(env.accidents) + ','
                            + str(count_shield) + ',' + str(current_time - start_time) + ',' +
                            str(current_time - initial_time) + ','
                            + str(corner) + ',' + str(mealy_size) + ',' + str(game_size)+','
                            + str(shield_min_depth)+','+ str( game_safe_states))
                        target_train.write("\n")
                        if next_episode_test:
                            current_episode_type = EPISODE_TYPE_TEST
                            self.training = False
                            next_episode_test = False
                        else:
                            episode += 1
                    elif current_episode_type == EPISODE_TYPE_TEST:
                        assert not self.training
                        target_test.write(
                            str(episode) + ',' + str(episode_reward) + ',' + str(episode_step) + ','
                            + str(self.step) + ',' + str(env.accidents) + ','
                            + str(count_shield) + ',' + str(current_time - start_time) + ',' +
                            str(current_time - initial_time) + ','
                            + str(corner) + ',' + str(mealy_size) + ',' + str(game_size)+','
                            + str(shield_min_depth)+','+ str( game_safe_states ))
                        target_test.write("\n")
                        current_episode_type = EPISODE_TYPE_TRAIN
                        self.training = True
                        episode += 1
                    else:
                        raise NotImplementedError('no episode type ' + current_episode_type)

                    target_train.close()
                    target_train = open(local_TRAIN_RESULT_FILE_NAME, 'a')
                    target_test.close()
                    target_test = open(local_TEST_RESULT_FILE_NAME, 'a')
                    observation = None
                    episode_step = None
                    episode_reward = None
                    if self.step > TOTAL_TRAINING_STEP:
                        # print('self.step > TOTAL_TRAINING_STEP, terminate')
                        target_train.close()
                        target_test.close()
                        break
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})



    def _on_test_begin(self):
        pass

    def _on_test_end(self):
        pass

    def reset_states(self):
        pass

    # def forward(self, observation):
    #     raise NotImplementedError()

    def backward(self, reward, terminal):
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        raise NotImplementedError()

    def load_weights(self, filepath):
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError()

    @property
    def metrics_names(self):
        return []


class AbstractDQNAgent(Agent):
    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
        super(AbstractDQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn(
                '`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(
                    delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class DQNAgent(AbstractDQNAgent):
    def __init__(self, model, enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError(
                'Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, self.nb_actions):
            raise ValueError(
                'Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(
                    model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output._keras_shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                     output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                     output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(input=model.input, output=outputlayer)

        # Related objects.
        self.model = model
        self.policy = BoltzmannQPolicy()
        self.test_policy = GreedyQPolicy()

        # State.
        self.reset_states()

    def get_config(self):
        config = super(DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(input=ins + [y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, observation, manual=False, banned_actions=None):
        # Select an action.
        if banned_actions is None:
            banned_actions = []
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        for action in banned_actions:
            q_values[action] = -1000
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)
        if manual:
            action = int(raw_input("action?\n"))
        if self.processor is not None:
            action = self.processor.process_action(action)

        if action in banned_actions:
            raise ValueError

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        metrics = [np.nan for _ in self.metrics_names]

        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal, training=self.training)

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()
        return metrics

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
