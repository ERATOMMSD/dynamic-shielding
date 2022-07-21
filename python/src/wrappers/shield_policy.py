import traceback
from types import MethodType
from typing import Optional, Union

import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

from src.wrappers.safe_padding_wrapper import SafePaddingWrapper
from src.wrappers.shield_wrappers import PreemptiveShieldWrapper


def shield_policy(model, eval_env: Optional[Union[PreemptiveShieldWrapper, SafePaddingWrapper]] = None,
                  force_eval: bool = False):
    assert eval_env is None or isinstance(eval_env, PreemptiveShieldWrapper) or \
           isinstance(eval_env, SafePaddingWrapper), "eval_env must be a PreemptiveShieldWrapper or SafePaddingWrapper"

    # This is the fixed method for the preemptive shields
    def preemptive_fixed(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        mean_actions = self.action_net(latent_pi)

        # HERE we access the environment and ask for the disabled_actions actions
        # The environment must be wrapped with PreemptiveShieldWrapper (gym.Wrapper)
        # - - - - - - - - - - - - - - - - - - - - - - - -
        for idx, env in enumerate(self.environments):
            # We do not use shield in the evaluation.
            if force_eval or any(map(lambda frame: 'stable_baselines3/common/evaluation.py' in frame.filename and
                                                   'evaluate_policy' in frame.name,
                                     traceback.extract_stack())):
                if eval_env is not None:
                    disabled = eval_env.get_shield_disabled_actions()
                else:
                    disabled = []
            else:
                disabled = env.get_shield_disabled_actions()
            if len(disabled) < len(mean_actions[idx]):
                for dis in disabled:
                    mean_actions[idx][dis] = float('-inf')
        # - - - - - - - - - - - - - - - - - - - - - - - -

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    model.policy.environments = model.env.envs
    model.policy._get_action_dist_from_latent = MethodType(preemptive_fixed, model.policy)
