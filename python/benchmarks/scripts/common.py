from enum import Enum, auto
from typing import Optional, Dict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FACTOR: float = 1.0


class Benchmarks(Enum):
    CAR_RACING = auto()
    CLIFFWALKING = auto()
    GRID_WORLD = auto()
    SELF_DRIVING_CAR = auto()
    SIDEWALK = auto()
    TAXI = auto()
    WATER_TANK = auto()


class Configurations(Enum):
    NO_SHIELD = auto()
    SAFE_PADDING = auto()
    DYNAMIC_SHIELD = auto()


CRASH_KEYS = {
    Benchmarks.CAR_RACING: 'total_spin_episodes',
    Benchmarks.CLIFFWALKING: 'stats/cliff',
    Benchmarks.GRID_WORLD: 'crash_episodes',
    Benchmarks.SELF_DRIVING_CAR: 'crash_episodes',
    Benchmarks.SIDEWALK: 'crash_episodes',
    Benchmarks.TAXI: 'stats/broken_count',
    Benchmarks.WATER_TANK: 'crash_episodes',
}

EPISODES_KEYS = {
    Benchmarks.CAR_RACING: 'episodes',
    Benchmarks.CLIFFWALKING: 'stats/episodes',
    Benchmarks.GRID_WORLD: 'episodes',
    Benchmarks.SELF_DRIVING_CAR: 'stats/episodes',
    Benchmarks.SIDEWALK: 'episodes',
    Benchmarks.TAXI: 'stats/episodes',
    Benchmarks.WATER_TANK: 'stats/episodes',
}

SUCCESS_RATE_KEYS = {
    Benchmarks.CAR_RACING: 'eval/success_rate',
    Benchmarks.CLIFFWALKING: 'eval/success_rate',
    Benchmarks.GRID_WORLD: 'eval/success_rate',
    Benchmarks.SELF_DRIVING_CAR: 'eval/success_rate',
    Benchmarks.SIDEWALK: 'eval/success_rate',
    Benchmarks.TAXI: 'eval/success_rate',
    Benchmarks.WATER_TANK: 'eval/success_rate',
}

WALL_TIME_KEYS = {
    Benchmarks.CAR_RACING: 'wall_time',
    Benchmarks.CLIFFWALKING: 'wall_time',
    Benchmarks.GRID_WORLD: 'wall_time',
    Benchmarks.SELF_DRIVING_CAR: 'wall_time',
    Benchmarks.SIDEWALK: 'wall_time',
    Benchmarks.TAXI: 'wall_time',
    Benchmarks.WATER_TANK: 'wall_time',
}


def get_scalars(logs, NUM_TIMESTEP: Optional[int] = None,
                MAX_SHIELD_LIFE: Optional[float] = None,
                LEARNING_RATE: Optional[float] = None) -> Dict:
    scalars = {}

    for path in logs:
        log = str(path)
        event_acc = EventAccumulator(log, size_guidance={'scalars': 0})
        event_acc.Reload()
        if ('rollout/ep_rew_mean' not in event_acc.Tags()['scalars']) or (
                #                'episodes' not in event_acc.Tags()['scalars']) or (
                'eval/mean_reward' not in event_acc.Tags()['scalars']):
            continue
        if 'eval/safe_rate' not in event_acc.Tags()['scalars']:
            continue
        scalars[log] = {}

        for tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(tag)
            scalars[log][tag] = [event.value for event in events]
        scalars[log]['step'] = [event.step for event in event_acc.Scalars('rollout/ep_rew_mean')]
        scalars[log]['wall_time'] = [event.wall_time for event in event_acc.Scalars('rollout/ep_rew_mean')]
        scalars[log]['eval/wall_time'] = [event.wall_time - scalars[log]['wall_time'][0] for event in
                                          event_acc.Scalars('eval/mean_reward')]

        if NUM_TIMESTEP is not None and \
                (scalars[log]['step'][-1] < NUM_TIMESTEP or scalars[log]['step'][-1] > NUM_TIMESTEP + int(1e4)):
            scalars.pop(log)
            continue
        if 'shield/factor' in scalars[log] and scalars[log]['shield/factor'][0] != FACTOR:
            scalars.pop(log)
            continue
        if MAX_SHIELD_LIFE is not None and 'shield/max_shield_life' in scalars[log] and \
                scalars[log]['shield/max_shield_life'][0] != MAX_SHIELD_LIFE:
            scalars.pop(log)
            continue
        if LEARNING_RATE is not None and abs(scalars[log]['train/learning_rate'][-1] - LEARNING_RATE) > 1e-6:
            scalars.pop(log)
            continue
        if 'skip_mealy_size' in scalars[log].keys() and scalars[log]['skip_mealy_size'][-1] != 0:
            scalars.pop(log)
            continue
    #        scalars[log]["episodes"] = [event.value for event in event_acc.Scalars('episodes') if
    #                                    event.step in scalars[log]['step']]
    #        if len(scalars[log]["episodes"]) != len(scalars[log]["rollout/ep_rew_mean"]):
    #            scalars.pop(log)
    #            continue
    return scalars
