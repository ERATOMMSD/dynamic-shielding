import numpy as np
from gym.envs.toy_text.cliffwalking import CliffWalkingEnv

from benchmarks.cliffwalking.cliffwalking_alphabet import EnvironmentActions, CliffWalkingInputOutputManager, \
    AgentActions, Observations


class CliffWalkingExt(CliffWalkingEnv):

    def __init__(self):
        super().__init__()
        self.cliff = 0
        self.goal = 0
        self.episodes_count = 0
        self.steps = 0
        self.explored_area = set()
        self.recent_exploration = []
        self.penalty = 0.0
        self.io_manager = CliffWalkingInputOutputManager()

    def set_penalty(self, penalty: float):
        assert penalty >= 0.0
        self.penalty = penalty

    def step(self, action: int):

        previous_state = self.s

        next_state, reward, done, info = super().step(action)

        info['p1_action'] = action
        info['p2_action'] = EnvironmentActions.IDLE.value
        info['output'] = self.output(previous_state, action, next_state)

        # Additional info for debugging
        info['previous_state'] = previous_state
        info['p1_action_name'] = AgentActions(action).name
        info['output_name'] = Observations(info['output']).name
        info['is_success'] = done
        info['is_crash'] = self.io_manager.is_cliff(info['output'])

        self.explored_area.add(self.decode(next_state))

        if done:
            self.goal += 1
        elif self.io_manager.is_cliff(info['output']):
            reward -= self.penalty
            self.cliff += 1
            done = True

        return next_state, reward, done, info

    def output(self, source: int, action: int, target: int) -> int:
        # valuation = [CLIFF if reward == -100, HIT_WALL if agent is in the same position, GOAL when is done]
        if self.is_cliff(source, action):
            return Observations.CLIFF.value
        elif self.is_goal(source, action):
            return Observations.GOAL.value
        else:
            return Observations.NORMAL.value

    def is_cliff(self, source: int, action: int) -> bool:
        pos_y, pos_x = self.decode(source)
        # shape assumed to be (4, 12)
        # start at position (y, x) = (3, 0)
        return ((pos_y == 2 and 0 < pos_x < 11 and action == AgentActions.DOWN.value) or
                (pos_y == 3 and pos_x == 0 and action == AgentActions.RIGHT.value) or
                (pos_y == 3 and pos_x == 11 and action == AgentActions.LEFT.value))

    def hit_wall(self, source: int, action: int) -> bool:
        pos_y, pos_x = self.decode(source)
        size_y, size_x = self.shape
        return ((pos_x == 0 and action == AgentActions.LEFT.value) or
                (pos_y == 0 and action == AgentActions.UP.value) or
                (pos_y == size_y - 1 and action == AgentActions.DOWN.value) or
                (pos_x == size_x - 1 and action == AgentActions.RIGHT.value))

    def is_goal(self, source: int, action: int) -> bool:
        """
        # shape is (4, 12)
        # W: wall, S: start, C: cliff, @: agent
            0                     11
          W W W W W W W W W W W W W W
        0 W                         W
        1 W                         W
        2 W                       @ W
        3 W S C C C C C C C C C C G W
          W W W W W W W W W W W W W W
        """
        pos_y, pos_x = self.decode(source)
        return pos_y == 2 and pos_x == 11 and action == AgentActions.DOWN.value

    def visited_cell(self, state):
        return self.decode(state) in self.explored_area

    def decode(self, s):
        """"
        Returns the position of the agent in that state
        """
        return np.unravel_index(s, self.shape)

    def reset(self):
        self.episodes_count += 1
        self.recent_exploration.append(len(self.explored_area))
        self.explored_area.clear()
        return super().reset()
