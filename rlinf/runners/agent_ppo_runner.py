# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous PPO scheduling for multi-turn agents."""

from typing import Any, Optional

from rlinf.data.schema.agent_trajectory import prepare_agent_ppo_targets
from rlinf.runners.agent_runner import AgentRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle


class AgentPPORunner(AgentRunner):
    """Complete both model updates and weight installation before each rollout."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        placement = self.component_placement
        actor_gpus = set(placement.get_hardware_ranks("actor"))
        critic_gpus = set(placement.get_hardware_ranks("critic"))
        rollout_gpus = set(placement.get_hardware_ranks("rollout"))
        self.shared_training = bool(actor_gpus & critic_gpus)
        self.release_rollout = bool(rollout_gpus & (actor_gpus | critic_gpus))

        self.rollout_output_channels = [self.rollout_channel]
        if not self.shared_training and self.recompute_logprobs:
            self.actor_rollout_channel = Channel.create("ActorRollout")
            self.rollout_output_channels.append(self.actor_rollout_channel)
        self.training_channel = Channel.create("PPOTraining")

    def init_rollout_workers(self) -> None:
        """Reuse agent initialization and convert the independent critic checkpoint."""
        super().init_rollout_workers()
        if (
            self.cfg.runner.resume_dir is None
            and self.cfg.critic.megatron.use_hf_ckpt
            and not self.cfg.critic.megatron.get("mbridge", False)
        ):
            from rlinf.utils.ckpt_convertor.megatron_convertor.convert_hf_to_mg import (
                convert_hf_to_mg,
            )

            convert_hf_to_mg(
                self.cfg.critic.megatron.ckpt_convertor.hf_model_path,
                self.cfg.critic.megatron.ckpt_convertor,
            )

    def _sync_weights(
        self, do_offload: bool = True, weight_version: Optional[int] = None
    ) -> None:
        if weight_version is None:
            weight_version = self.global_steps
        if self.component_placement.is_collocated:
            super()._sync_weights(do_offload=do_offload, weight_version=weight_version)
            return

        super()._sync_weights(do_offload=False, weight_version=weight_version)
        if not do_offload:
            return
        if self.cfg.actor.offload_weight or self.cfg.actor.offload_grad:
            self.actor.offload_model_weights_and_grad(
                offload_weight=self.cfg.actor.offload_weight,
                offload_grad=self.cfg.actor.offload_grad,
            ).wait()
        if self.cfg.actor.offload_optimizer:
            self.actor.offload_megatron_optimizer().wait()

    def _start_actor_update(
        self, input_channel: Channel, version: Optional[int] = None
    ) -> tuple[Handle, Optional[Handle]]:
        """Freeze actor logprobs before updating, preserving behavior metadata."""
        inference_handle = None
        if self.recompute_logprobs:
            inference_handle = self.actor.run_inference(
                input_channel=input_channel,
                output_channel=self.actor_inference_channel,
                compute_ref_logprobs=False,
            )
            inference_handle.wait()
            input_channel = self.actor_inference_channel
        return (
            self.actor.run_training(input_channel=input_channel, version=version),
            inference_handle,
        )

    def _run_training(
        self, rollout_handle: Handle
    ) -> tuple[Optional[dict], list[dict], dict]:
        """Prepare fixed targets and update models on shared or separate devices."""
        if self.cfg.algorithm.adv_type == "gae_dynamic":
            return self._run_critic_owned_training(rollout_handle)
        # Old predictions: independent devices can infer in parallel.
        critic_infer_handle: Handle = self.critic.run_inference(
            input_channel=self.rollout_channel,
            output_channel=self.value_channel,
            do_offload=False,
        )
        actor_infer_handle = None
        if not self.shared_training and self.recompute_logprobs:
            actor_infer_handle: Handle = self.actor.run_inference(
                input_channel=self.actor_rollout_channel,
                output_channel=self.actor_inference_channel,
                compute_ref_logprobs=False,
            )
            actor_infer_handle.wait()
        critic_infer_handle.wait()

        # GAE: fix globally normalized targets for every minibatch.
        with self.timer("advantages"):
            prepare_agent_ppo_targets(
                input_channel=self.value_channel,
                output_channel=[
                    self.training_channel,
                    self.critic_output_channel,
                ],
                batch_size=self.cfg.data.rollout_batch_size,
                gamma=self.cfg.algorithm.gamma,
                gae_lambda=self.cfg.algorithm.gae_lambda,
                normalize_advantages=self.cfg.algorithm.normalize_advantages,
                logprob_channel=(
                    None
                    if self.shared_training or not self.recompute_logprobs
                    else self.actor_inference_channel
                ),
            )

        # Updates: serialize shared devices; otherwise train in parallel.
        critic_handle: Handle = self.critic.run_training(
            input_channel=self.critic_output_channel,
            compute_rollout_metrics=False,
        )
        if not self.shared_training:
            actor_handle: Handle = self.actor.run_training(
                input_channel=self.training_channel,
            )
        critic_metrics = critic_handle.wait()
        critic_training_metrics = critic_metrics[0][1]

        if self.shared_training:
            actor_handle, actor_infer_handle = self._start_actor_update(
                self.training_channel,
            )
        actor_metrics = actor_handle.wait()
        actor_rollout_metrics = actor_metrics[0][0]
        actor_training_metrics = actor_metrics[0][1]

        training_metrics = self._training_metrics(
            actor_training_metrics, critic_training_metrics
        )
        durations = {
            "training": actor_handle.consume_duration(),
            "critic/training": critic_handle.consume_duration(),
            "critic/inference": critic_infer_handle.consume_duration(
                reduction_type="min"
            ),
        }
        if actor_infer_handle is not None:
            durations["inference"] = actor_infer_handle.consume_duration(
                reduction_type="min"
            )
        return actor_rollout_metrics, training_metrics, durations

    def _run_critic_owned_training(
        self, rollout_handle: Handle
    ) -> tuple[Optional[dict], list[dict], dict]:
        """Compute fixed critic targets, update the critic, then update the actor."""
        critic_infer_handle = self.critic.run_inference(
            input_channel=self.rollout_channel,
            output_channel=self.value_channel,
            do_offload=False,
        )
        critic_infer_handle.wait()
        critic_handle = self.critic.run_training(
            input_channel=self.value_channel,
            output_channel=self.critic_output_channel,
            compute_rollout_metrics=False,
        )
        critic_training_metrics = critic_handle.wait()[0][1]

        actor_handle, actor_infer_handle = self._start_actor_update(
            self.critic_output_channel
        )
        rollout_metrics, training_metrics = actor_handle.wait()[0]
        training_metrics = self._training_metrics(
            training_metrics, critic_training_metrics
        )
        durations = {
            "training": actor_handle.consume_duration(),
            "critic/training": critic_handle.consume_duration(),
            "critic/inference": critic_infer_handle.consume_duration(
                reduction_type="min"
            ),
        }
        if actor_infer_handle is not None:
            durations["inference"] = actor_infer_handle.consume_duration(
                reduction_type="min"
            )
        return rollout_metrics, training_metrics, durations

    def _training_metrics(
        self, actor_metrics: list[dict], critic_metrics: list[dict]
    ) -> list[dict]:
        """Report each minibatch and its critic metrics from the last update epoch.

        Critic epochs can outnumber actor epochs. Keep the same logging unit as
        the agent runner instead of pairing optimizer updates one for one.
        """
        minibatches = self.cfg.algorithm.n_minibatches
        return [
            {**actor, **critic}
            for actor, critic in zip(
                actor_metrics, critic_metrics[-minibatches:], strict=True
            )
        ]
