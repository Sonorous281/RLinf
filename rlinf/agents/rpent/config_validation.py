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

"""Validate the example's simulator and learner contracts before scheduling."""

from omegaconf import OmegaConf


def validate_rpent_cfg(cfg):
    env = cfg.tools.rpent
    if env.workflow not in ("episode", "libero", "libero_rl"):
        raise ValueError("Unknown RPent workflow; use episode, libero_rl or libero")
    if (
        not env.env_gpus
        or min(env.episodes_per_worker, env.max_env_initializations_per_worker) < 1
    ):
        raise ValueError("RPent needs a render GPU and at least one simulator slot")
    if env.keep_episodes not in ("all", "failures", "none"):
        raise ValueError("keep_episodes must be all, failures or none")
    if not OmegaConf.select(cfg, "session_server.enable"):
        raise ValueError("RPent training requires session_server.enable")
    if cfg.session_server.get("verify_assembly", False):
        raise ValueError("RPent uses the chat route with server-side assembly")
    if (
        env.workflow in ("episode", "libero_rl")
        and not cfg.rollout.sglang.enable_shared_multi_modal_processor
    ):
        raise ValueError(
            "RPent image observations require enable_shared_multi_modal_processor"
        )
    if (
        not 0
        < cfg.agentloop.minimum_output_tokens
        <= cfg.algorithm.sampling_params.max_new_tokens
    ):
        raise ValueError("Invalid output token budget for RPent")
    if cfg.runner.synchronous:
        if (
            cfg.critic.use_critic_model
            or cfg.algorithm.adv_type != "grpo_dynamic"
            or cfg.data.sampling_batch_size < cfg.data.rollout_batch_size
        ):
            raise ValueError(
                "Synchronous RPent uses GRPO without a critic, with at least one candidate per admitted group"
            )
    elif not (
        cfg.runner.async_mode
        and cfg.critic.use_critic_model
        and cfg.algorithm.adv_type == "sao_gae"
        and cfg.algorithm.loss_type == "dis"
        and cfg.critic.compute_advantages_after_update
    ):
        raise ValueError(
            "Asynchronous RPent uses SAO and recomputes advantages after the critic update"
        )
    if cfg.rollout.pause_mode != "retract":
        raise ValueError(
            "Use retract to invalidate the rollout KV cache when publishing new weights"
        )
    if cfg.algorithm.recompute_logprobs and not (
        cfg.rollout.return_logprobs
        and cfg.algorithm.importance_sampling_fix
        and cfg.algorithm.importance_sampling_clip > 0
    ):
        raise ValueError(
            "Actor recomputation needs rollout logprobs and a positive TIS clip"
        )
    if cfg.rollout_buffer.max_buffered_groups < cfg.data.rollout_batch_size:
        raise ValueError("The rollout buffer must hold at least one update")
