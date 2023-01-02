import torch
import copy
from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.nav import (
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import CategoricalNet
from habitat_baselines.rl.ppo.policy import Policy

class MultiStepPolicy(Policy):
    def __init__(self, net, dim_actions, no_critic=False, multi_step_cfg=None):
        super().__init__(net, dim_actions, no_critic=False)
        self.multi_step_cfg = multi_step_cfg
        self.action_rnn_hidden_states = None
        self.predicted_steps = self.multi_step_cfg.predicted_steps

    def forward_eval(self, observations, rnn_hidden_states, prev_actions, masks):
        distribution = None
        hist_actions = []
        if len(prev_actions.size()) == 3:
            prev_actions = prev_actions.contiguous().view(-1, prev_actions.size(2))
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        distribution = self.action_distribution(features)
        return distribution.logits, rnn_hidden_states

    def forward_train(self, observations, rnn_hidden_states, prev_actions, masks):
        T, N = prev_actions.shape[0], prev_actions.shape[1]
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks.clone(), is_sge_retain=False
        )
        distribution = self.action_distribution(features)
        hist_actions = [distribution.logits.view(T, N, -1).unsqueeze(0)]

        for _ in range(1, self.predicted_steps):
            prev_actions = distribution.logits.view(T, N, -1).max(dim=-1).indices.unsqueeze(-1)
            features, rnn_hidden_states = self.net(
                observations, rnn_hidden_states, prev_actions, masks.clone(), is_sge_retain=True
            )
            distribution = self.action_distribution(features)
            hist_actions.append(distribution.logits.view(T, N, -1).unsqueeze(0))
        
        return torch.cat(hist_actions).permute(1, 2, 0, 3), rnn_hidden_states

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, is_training=True):
        if is_training:
            return self.forward_train(observations, rnn_hidden_states, prev_actions, masks)
        else:
            return self.forward_eval(observations, rnn_hidden_states, prev_actions, masks)
