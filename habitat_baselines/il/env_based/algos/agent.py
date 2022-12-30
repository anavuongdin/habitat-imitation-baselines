#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat import logger
from habitat.utils import profiling_wrapper

from habitat_baselines.il.env_based.common.multi_step_conversion import convert_multi_step_actions


class ILAgent(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_envs: int,
        num_mini_batch: int,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        multi_step_cfg: Optional[dict] = None,
    ) -> None:

        super().__init__()

        self.model = model

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.predicted_steps = multi_step_cfg.predicted_steps

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(model.parameters()).device

    def forward(self, *x):
        raise NotImplementedError
    
    def _get_multi_actions_batch(self, data_generator):
        pass

    def update(self, rollouts) -> Tuple[float, float, float]:
        total_loss_epoch = 0.0

        profiling_wrapper.range_push("BC.update epoch")
        data_generator = rollouts.recurrent_generator(
            self.num_mini_batch
        )
        cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=- 100, reduction="none")
        hidden_states = []

        for sample in data_generator:
            (
                obs_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                masks_batch,
                idx
            ) = sample
            multi_step_actions = convert_multi_step_actions(actions_batch, self.predicted_steps)

            # Reshape to do in a single forward pass for all steps
            (
                multi_step_predictions,
                rnn_hidden_states,
            ) = self.model(
                obs_batch,
                recurrent_hidden_states_batch,
                prev_actions_batch,
                masks_batch,
            )

            action_loss = cross_entropy_loss(multi_step_predictions.permute(0, 3, 1, 2), multi_step_actions.squeeze(-1).long()).sum(dim=-1)
            self.optimizer.zero_grad()
            inflections_batch = obs_batch["inflection_weight"]

            total_loss = ((inflections_batch * action_loss).sum(0) / inflections_batch.sum(0)).mean()

            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            total_loss_epoch += total_loss.item()
            hidden_states.append(rnn_hidden_states)

        profiling_wrapper.range_pop()

        hidden_states = torch.cat(hidden_states, dim=1)

        total_loss_epoch /= self.num_mini_batch

        return total_loss_epoch, hidden_states

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass


EPS_PPO = 1e-5


class DecentralizedDistributedMixin:

    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self.model, self.device)  # type: ignore
        # self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss: Tensor) -> None:
        super().before_backward(loss)  # type: ignore

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])  # type: ignore
        else:
            self.reducer.prepare_for_backward([])  # type: ignore


class DDPILAgent(DecentralizedDistributedMixin, ILAgent):
    pass
