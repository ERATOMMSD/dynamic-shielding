from typing import Dict, Set, List, Tuple

from src.model import SafetyGame


# This is an easy function. We can add this in safety_game.py after discussing about types.

def solve_game(game) -> Tuple[Set[int], Dict[int, List[int]]]:
    #  Is transition_function a total function? I assume that it is
    #  http://www.lsv.fr/~dwb/graph-games-pi.pdf page 6
    #  https://pdfs.semanticscholar.org/a0cb/c864a25e0c2c8e90f483c31b59a7cf63d735.pdf page 36
    win_set: Set = set(game.safeStates)  # game.acceptingStates is the set of safe states (?)
    # win_set: np.ndarray = np.ndarray(game.safeStates)
    win_strategy: Dict[int, List[int]]  # (set of states: Int) --> (list of actions: List[int])
    action_1 = game.getPlayer1Alphabet()  # angelic player, controller, shield, etc.
    action_2 = game.getPlayer2Alphabet()  # demonic player, environment, non-determinism of the system, etc.
    while True:
        # print('win_set', win_set)
        win_strategy = {}
        # pre_win_set will be the set states that are still safe after one transition, i.e.,
        # pre_win_set = {q' | \exists action a of player 1, \for_all action b of player 1, q' --(a, b)--> win_set }
        pre_win_set = set()
        for q in win_set:
            for a in action_1:
                if (q, a) in game.power_transitions:
                    post_q_a = game.power_transitions[q, a]
                    # If, by choosing 'a', player 1 can control the game to be in win_set after one transition,
                    # then, we add q in pre_win_set
                    if len(post_q_a) > 0 and post_q_a.issubset(win_set):
                        pre_win_set.add(q)
                        if q in win_strategy:
                            win_strategy[q].append(a)
                        else:
                            win_strategy[q] = [a]
        if win_set == pre_win_set:
            break  # player 1 can control the game to be in the set win_set == pre_win_set for any number of transitions
        win_set = pre_win_set
    return win_set, win_strategy


def test_dummy():
    # build a dummy game, for testing only
    dummy_actions_a = [1, 2]
    dummy_actions_b = [3, 4]
    dummy_game = SafetyGame(dummy_actions_a, dummy_actions_b)
    dummy_game.addSafeState(1)
    dummy_game.addSafeState(2)
    dummy_game.addSafeState(3)
    # Is transition_function a total function? <<---- NO!!
    dummy_game.add_transition(1, 1, 3, 2)
    # dummy_game.addTransition(1, 1, 4, 2)
    # dummy_game.addTransition(1, 2, 3, 2)
    dummy_game.add_transition(1, 2, 4, 3)
    dummy_game.add_transition(2, 1, 3, 2)
    dummy_game.add_transition(2, 1, 4, 2)
    # dummy_game.addTransition(2, 2, 3, 2)
    dummy_game.add_transition(2, 2, 4, 1)
    dummy_game.add_transition(3, 1, 3, 3)
    dummy_game.add_transition(3, 1, 4, 4)
    # dummy_game.addTransition(3, 2, 3, 4)
    # dummy_game.addTransition(3, 2, 4, 4)
    # dummy_game.addTransition(4, 1, 3, 1)
    # dummy_game.addTransition(4, 1, 4, 1)
    # dummy_game.addTransition(4, 2, 3, 2)
    # dummy_game.addTransition(4, 2, 4, 2)
    print('transitions =', dummy_game.transitions)
    # solve the game
    win_set, win_strategy = solve_game(dummy_game)
    print('set of winning states =', win_set)
    print('winning strategy =', win_strategy)


if __name__ == '__main__':
    test_dummy()
