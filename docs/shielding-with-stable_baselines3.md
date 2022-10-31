# Preemptive Shielding with Stable Baselines3

This document presents how we use a preemptive shield in reinforcement learning with [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). The following works for [`ActorCriticPolicy`](https://github.com/DLR-RM/stable-baselines3/blob/v1.2.0/stable_baselines3/common/policies.py#L367) (such as A2C and PPO). We tested this with Stable Baselines3 version 1.2.0. Since we modify a protected method, the following may not work with a newer version of Stable Baselines3.

The following files are particularly relevant to this topic.

- `src/wrappers/shield_policy.py`
- `benchmarks/common/train.py`
- `benchmarks/common/evaluate.py`

## Prediction by `ActorCriticPolicy` in Stable Baselines3

When an action is predicted with [`ActorCriticPolicy`](https://github.com/DLR-RM/stable-baselines3/blob/v1.2.0/stable_baselines3/common/policies.py#L367) for learning, evaluation, and actual use, there are the following steps. As you can see in their names, they are protected methods, and the API may change.

1. The latent code is obtained from the observation ([`_get_latent`](https://github.com/DLR-RM/stable-baselines3/blob/v1.2.0/stable_baselines3/common/policies.py#L594)).
2. An action distribution is created from the latent code ([`_get_action_dist_from_latent`](https://github.com/DLR-RM/stable-baselines3/blob/v1.2.0/stable_baselines3/common/policies.py#L613)).
3. An action is sampled from the distribution ([`_predict_`](https://github.com/DLR-RM/stable-baselines3/blob/v1.2.0/stable_baselines3/common/policies.py#L639)).

## Enforcement of a Shield

Among the above three steps, the creation of the action distribution is the part modified by a shield so that unsafe actions are not sampled. Namely, we modify the weight of unsafe actions to be `-inf` so that they are not sampled.

[`shield_policy`](https://github.com/ERATOMMSD/dynamic-shielding/blob/ATVA2022/python/src/wrappers/shield_policy.py#L19) is the function to change the policy of the given model so that the action distribution is created with the above modification. After making `mean_actions` from `latent_pi`, for each action, we check if the shield forbids it. For each forbidden action, we change the weight in `mean_actions` to `-inf` so that they are not sampled. Finally, we return the action distribution using the modified `mean_actions`.

We note that when we use the model for evaluation, the environment `eval_env` dedicated to it is used. This is to prevent some issues due to the concurrent use of one dynamic shield for evaluation and training simultaneously.

## Use of `shield_policy`

When using `shield_policy` for training, you must call `shield_policy` before starting the training. [`train.py`](https://github.com/ERATOMMSD/dynamic-shielding/blob/ATVA2022/python/benchmarks/common/train.py#L275) shows an example. We note that the model must be created from an environment wrapped with a preemptive shield.

When using `shield_policy` for evaluation, you must call `shield_policy` before the prediction. [`evaluate.py`](https://github.com/ERATOMMSD/dynamic-shielding/blob/ATVA2022/python/benchmarks/common/evaluate.py) shows an example.
