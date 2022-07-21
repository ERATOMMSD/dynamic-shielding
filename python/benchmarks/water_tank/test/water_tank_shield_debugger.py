import random
import logging as log
from benchmarks.water_tank.water_tank_arena import make_water_tank_mdp, AgentActions, EnvironmentActions, WaterTankInputOutputManager
from src.exceptions.shielding_exceptions import UnsafeStateError, InvalidMoveError, InvalidMoveWithShieldError


class ShieldedDebuggerWaterTank(object):
    def __init__(self, initial_level, capacity, shield):
        self.arena = make_water_tank_mdp(initial_level, capacity)
        self.arena_state = self.arena.getInitialState()
        self.shield = shield
        self.shield_enabled = self.shield.current_is_winning()
        self.shield_at_start = self.shield_enabled
        self.io_manager = WaterTankInputOutputManager()
        self.error_status = False
        self.output_history = []
        self.actions_history = []
        self.logged_failures = []

    def reset(self, trial=None, debug=False):
        if debug:
            log.debug(f'Trial {trial}')
            log.debug(f"[Shield at start {self.shield_at_start}]")
            log.debug(f"[Shield at the end {self.shield_enabled}]")
            log.debug(f"Shield state: {self.shield.state}")
            log.debug(f'Number of steps: {len(self.output_history)}')
            log.debug(f'Last movements {list(map(lambda x: x[0], self.actions_history[-6:]))}')
            log.debug(f'Last outputs {self.output_history[-6:]}')
            log.debug(f'Water level at the end: {self.arena_state}')
            log.debug("--------------------------------------------------------------")
            self.logged_failures.append((trial, self.shield.state))
        self.arena_state = self.arena.getInitialState()
        self.shield.reset()
        self.shield_enabled = self.shield.current_is_winning()
        self.shield_at_start = self.shield_enabled
        self.error_status = False
        self.output_history = []
        self.actions_history = []

    def _change_is_possible(self):
        """"
            A change is possible after keeping the water tank in the same status for three turns.
            The last three ticks were same or there was never a change.
        """
        return ((len(self.output_history) > 2 and
                 len(set(map(lambda x: self.io_manager.is_open(x), self.output_history[-3:]))) == 1) or
                len(set(map(lambda x: self.io_manager.is_open(x), self.output_history))) == 1)

    def valid_move(self, p1_action):
        """"
        Anything is valid in the beginning or
        action is same with previous one or
        we are allowed to change action.
        """
        return (len(self.output_history) == 0 or
                self.actions_history[-1][0] == p1_action or
                self._change_is_possible())

    def disable_shield(self):
        self.shield_enabled = False

    def move(self, p1_action: int, p2_action: int = None):
        if p2_action is None:
            p2_candidates = self.arena.getProbabilisticSuccessor(self.arena_state, p1_action)
            # Weights encodes the probabilities of the transition
            [(p2_action, prob, successor)] = random.choices(p2_candidates,
                                                            weights=list(map(lambda x: x[1], p2_candidates)), k=1)

        valid_movement = self.valid_move(p1_action)
        output = self.arena.getOutput(self.arena_state, p1_action, p2_action)
        self.output_history.append(output)
        self.actions_history.append((p1_action, p2_action))

        prob, self.arena_state = self.arena.getSuccessorWithP2Action(self.arena_state, p1_action, p2_action)
        self.shield.move(p1_action, p2_action, output)

        # Stopping the simulation when it violates the safety specification
        # TODO: Shall the output still be the normal output or error?
        if not valid_movement:
            if self.shield_enabled:
                raise InvalidMoveWithShieldError
            raise InvalidMoveError

        log.info(f'Agent action {p1_action}, Environment: {p2_action}')
        self.error_status = self.io_manager.is_error(output)
        log.info('Output:', str(output), "[Error:{0}]".format(self.error_status))
        log.info("--------------------------------------------------------------")
        if self.error_status:
            raise UnsafeStateError

    def reconstruct_shield(self):
        self.shield.reconstructShield()

    def preemptive(self):
        suggestions = [str(AgentActions.CLOSE), str(AgentActions.OPEN)]
        if self.shield_enabled:
            try:
                suggestions = self.shield.preemptive()
                log.info('Shield suggests:', suggestions)
            except UnsafeStateError:
                self.shield_enabled = False
                "This unsafe error was raised by the shield, but might not be really unsafe."
                if self.error_status:
                    raise UnsafeStateError
        else:
            log.info('Shield disabled:', suggestions)
        return suggestions

    def current_is_winning(self):
        result = self.shield.current_is_winning()
        return result

    def moves(self, moves):
        for p1_action, p2_action in moves:
            self.move(p1_action.value, p2_action.value)

    def explore_all(self, history=None, unsafe_states=False):
        if not history:
            history = []

        for p1_action, p2_action in [(AgentActions.OPEN, EnvironmentActions.NONE),
                                     (AgentActions.OPEN, EnvironmentActions.NORMAL),
                                     (AgentActions.OPEN, EnvironmentActions.HIGH),
                                     (AgentActions.CLOSE, EnvironmentActions.NONE),
                                     (AgentActions.CLOSE, EnvironmentActions.NORMAL)]:
            self.moves(history)
            try:
                self.move(p1_action.value, p2_action.value)
            except UnsafeStateError as e:
                # Raise the exception when there shouldn't be unsafe states.
                log.info(f'Unsafe state after action {p1_action} - {p2_action}')
                if not unsafe_states:
                    raise e
            self.reset()

    def random_exploration(self, trials: int, steps: int):
        invalid_with_shield = [0] * trials
        unsafe_moves = [0] * trials
        invalid_without_shield = [0] * trials
        lengths = [0] * trials
        shield = [0] * trials
        # number of simulations from the initial state
        for trial in range(trials):
            self.reset()
            shield[trial] = 1 if self.shield_enabled else 0
            # length of the trace
            for step in range(steps):
                log.info(f'Step: {step}')
                suggestions = self.preemptive()
                p1_action = random.choice(suggestions)
                try:
                    self.move(p1_action)
                except UnsafeStateError:
                    unsafe_moves[trial] += 1
                    log.info(f"Trial {trial}: Unsafe move.")
                    break
                except InvalidMoveError:
                    invalid_without_shield[trial] += 1
                    log.info(f"Trial {trial}: Invalid move when shield disabled.")
                    break
                except InvalidMoveWithShieldError:
                    invalid_with_shield[trial] += 1
                    log.info(f"Trial {trial}: Invalid move when shield enabled.")
                    break
                lengths[trial] += 1
        self.reset()
        return lengths, shield, unsafe_moves, invalid_without_shield, invalid_with_shield

    def explore_all_traces(self, max_length):
        histories = [tuple()]
        self.reset()
        while histories:
            history = histories.pop()
            for p1_action, p2_action in [(AgentActions.OPEN, EnvironmentActions.NONE),
                                         (AgentActions.OPEN, EnvironmentActions.NORMAL),
                                         (AgentActions.OPEN, EnvironmentActions.HIGH),
                                         (AgentActions.CLOSE, EnvironmentActions.NONE),
                                         (AgentActions.CLOSE, EnvironmentActions.NORMAL)]:
                self.moves(history)
                try:
                    self.move(p1_action.value, p2_action.value)
                    if len(history) < max_length:
                        histories.append(tuple(history + tuple([(p1_action, p2_action)])))
                except (UnsafeStateError, InvalidMoveWithShieldError, InvalidMoveError):
                    # when an action is unsafe or the movement is invalid, we do not continue exploring the trace
                    pass
                self.reset()


