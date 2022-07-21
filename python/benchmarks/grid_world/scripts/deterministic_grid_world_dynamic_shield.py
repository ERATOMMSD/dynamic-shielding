import argparse
import os.path as osp
import sys

from py4j.java_gateway import JavaGateway, CallbackServerParameters

# Add the root of this project to the path
PROJECT_ROOT = osp.join(osp.dirname(__file__), '../../../')
sys.path.append(PROJECT_ROOT)
LOG_ROOT = osp.join(osp.dirname(__file__), '../logs/')

from benchmarks.grid_world.deterministic_grid_world_dynamic_shield import GridWorldDynamicShield
from benchmarks.grid_world.grid_world_specifications import safe, safe_with_no_crash_duration, safe2, no_wall
from benchmarks.grid_world.scripts.helper import configure_logger, configure_env


def grid_world_dynamic_shield(task_string: str, max_step: int, num_timesteps: int, num: int, lr: float, gamma: float,
                              min_depth: int, concurrent_reconstruction=False, max_shield_life: int = 100):
    # string constants for this experiment setting
    if concurrent_reconstruction:
        experiment_string = 'deepq_dynamic_shield_concurrent'
    else:
        experiment_string = 'deepq_dynamic_shield'
    algorithm = 'deepq_shielding'
    # configure the training environment
    configure_logger(
        osp.join(LOG_ROOT, f'{task_string}_{experiment_string}-{max_shield_life}-{min_depth}-{num_timesteps}-{num}/'))
    learn, env, env_type, env_id, alg_kwargs = configure_env(algorithm, f'benchmarks.grid_world:{task_string}')
    env.envs[0].env.MAX_STEP = max_step
    print('Training {} on {}:{} with arguments \n{}'.format(algorithm, env_type, env_id, alg_kwargs))

    # Set up py4j gateway
    gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
    shield = GridWorldDynamicShield(
        [safe, safe_with_no_crash_duration(5), safe_with_no_crash_duration(3), safe_with_no_crash_duration(1), safe2,
         no_wall], gateway,
        min_depth=min_depth,
        concurrent_reconstruction=concurrent_reconstruction,
        max_shield_life=max_shield_life)

    # train the model
    model = learn(
        env=env,
        seed=None,
        total_timesteps=num_timesteps,
        shield=shield,
        lr=lr,
        gamma=gamma,
        batch_size=128,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        target_network_update_freq=500,
        **alg_kwargs
    )
    # Save the trained model
    model.save(osp.join(LOG_ROOT, 'model'))
    gateway.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the experiment on grid world with DynamicShielding in Java')
    parser.add_argument('--task_string', type=str, default='grid_world2-v1',
                        help='the string repesentation of the task')
    parser.add_argument('--max_step', type=int, default=100,
                        help='the maximum step in one episode')
    parser.add_argument('--total_steps', type=int, default=int(1e5),
                        help='the number of steps in this experiment')
    parser.add_argument('--min_depth', type=int, default=0,
                        help='the minimum depth to be merged in the strong RPNI-style algorithm')
    parser.add_argument('--num', type=int, default=3,
                        help='the number to distinguish this execution')
    parser.add_argument('--concurrent_reconstruction', type=bool, default=True,
                        help='reconstruct shielding concurrently')
    parser.add_argument('--max_shield_life', type=int, default=100,
                        help='maximum episodes to refresh the learned shield')
    parser.add_argument('--lr', type=float, default=7e-3,
                        help='the learning rate')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='gamma')
    args = parser.parse_args()
    grid_world_dynamic_shield(args.task_string, args.max_step, args.num_timesteps, args.num, args.lr, args.gamma,
                              args.min_depth, args.concurrent_reconstruction, args.max_shield_life)
