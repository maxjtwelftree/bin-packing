import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Custom network architecture
        self.net_arch = [dict(pi=[256, 256], vf=[256, 256])]

        # Build the network
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        # Build the networks using the specified architecture
        super(CustomPolicy, self)._build(lr_schedule)
