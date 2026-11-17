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

import math

from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.agents.rpent.placement import RPentComponentPlacement
from rlinf.config import (
    shared_multi_modal_processor_enabled,
    validate_async_agent_cfg,
)
from rlinf.scheduler import Cluster


def validate_rpent_cfg(cfg: DictConfig) -> DictConfig:
    """Validate RPent planner PPO or evaluation, selected by ``tools.rpent``."""
    rpent = cfg.get("tools", {}).get("rpent")
    if rpent is None:
        return cfg

    if not rpent.env_gpus or rpent.episodes_per_worker < 1:
        raise ValueError(
            "tools.rpent needs at least one env GPU and one episode per worker, got "
            f"env_gpus={list(rpent.env_gpus)} and "
            f"episodes_per_worker={rpent.episodes_per_worker}."
        )
    initializations = rpent.get("max_env_initializations_per_worker", 8)
    if type(initializations) is not int or initializations < 1:
        raise ValueError(
            "tools.rpent.max_env_initializations_per_worker must be a positive "
            f"integer, got {initializations!r}."
        )
    if rpent.keep_episodes not in ("all", "failures", "none"):
        raise ValueError(
            "tools.rpent.keep_episodes must be all, failures or none, got "
            f"{rpent.keep_episodes!r}."
        )
    if rpent.get("store_images_on_disk", False) and rpent.keep_episodes != "all":
        raise ValueError(
            "tools.rpent.store_images_on_disk requires keep_episodes=all; "
            "training and checkpoint recovery still reference episode images."
        )
    if shared_multi_modal_processor_enabled(OmegaConf.select(cfg, "rollout.sglang")):
        if cfg.rollout.rollout_backend != "sglang" or not rpent.get(
            "store_images_on_disk", False
        ):
            raise ValueError(
                "Shared multimodal preprocessing for RPent requires SGLang and store_images_on_disk=True"
            )
        if cfg.rollout.get("env_vars", {}).get("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"):
            raise ValueError(
                "Remove SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE when enabling shared multimodal preprocessing"
            )
        learner_roles = (
            ("actor", "critic") if cfg.runner.task_type == "reasoning" else ()
        )
        for role in learner_roles:
            worker = cfg[role]
            if (
                worker.training_backend != "megatron"
                or not worker.megatron.get("mbridge", False)
                or worker.model.model_type != "qwen3_5"
            ):
                raise ValueError(
                    "Shared multimodal training currently requires Qwen3.5 Megatron-Bridge actor/critic"
                )
            if (
                worker.model.pipeline_model_parallel_size != 1
                or worker.model.get("context_parallel_size", 1) != 1
            ):
                raise ValueError(
                    "Shared multimodal position verification currently requires PP=1 and CP=1"
                )
    # Resolve legacy RPent budgets before the generic learner config is validated.
    for role in ("actor", "critic"):
        if cfg.get(role) is None:
            continue
        for field in ("cache_bytes", "prefetch_batches", "prefetch_bytes"):
            legacy = f"training_image_{field}"
            if legacy not in rpent:
                continue
            target = f"{role}.multimodal.{field}"
            multimodal = cfg[role].get("multimodal") or {}
            if field in multimodal and multimodal[field] != rpent[legacy]:
                raise ValueError(f"{target} conflicts with tools.rpent.{legacy}.")
            if field not in multimodal:
                with open_dict(cfg):
                    OmegaConf.update(cfg, target, rpent[legacy], force_add=True)
    agentloop = cfg.agentloop
    thinking_budget = agentloop.get("thinking_token_budget")
    if thinking_budget is not None:
        if type(thinking_budget) is not int or thinking_budget < 0:
            raise ValueError(
                "agentloop.thinking_token_budget must be null or a nonnegative integer, "
                f"got {thinking_budget!r}."
            )
        if not cfg.rollout.sglang.get("enable_custom_logit_processor", False):
            raise ValueError(
                "agentloop.thinking_token_budget requires "
                "rollout.sglang.enable_custom_logit_processor=True."
            )
        if not (agentloop.get("chat_template_kwargs") or {}).get(
            "enable_thinking", True
        ):
            raise ValueError("A thinking budget requires enable_thinking=True.")
        if not cfg.rollout.sglang.get("disable_overlap_schedule", False):
            raise ValueError(
                "agentloop.thinking_token_budget requires "
                "rollout.sglang.disable_overlap_schedule=True; "
                "overlapped decoding can expose stale CPU token history."
            )
    if (
        agentloop.toolcall_parser != "qwen3.5"
        or agentloop.agentic_tokenizer != "qwen3_5"
    ):
        raise ValueError(
            "RPent renders and parses Qwen3.5 chats: set agentloop.toolcall_parser="
            "qwen3.5 and agentloop.agentic_tokenizer=qwen3_5, got "
            f"{agentloop.toolcall_parser} and {agentloop.agentic_tokenizer}."
        )
    max_new_tokens = cfg.algorithm.sampling_params.max_new_tokens
    if not 0 < agentloop.minimum_output_tokens <= max_new_tokens:
        raise ValueError(
            "agentloop.minimum_output_tokens must be between 1 and "
            f"sampling_params.max_new_tokens={max_new_tokens}, "
            f"got {agentloop.minimum_output_tokens}."
        )
    if agentloop.max_infra_retries < 0 or not 0 <= agentloop.max_infra_retry_rate <= 1:
        raise ValueError(
            "agentloop.max_infra_retries must be nonnegative and "
            "agentloop.max_infra_retry_rate between 0 and 1, got "
            f"{agentloop.max_infra_retries} and {agentloop.max_infra_retry_rate}."
        )

    workflow = rpent.get("workflow", "libero")
    if workflow not in ("libero", "libero_rl"):
        raise ValueError(
            f"tools.rpent.workflow must be libero or libero_rl, got {workflow!r}."
        )
    if workflow == "libero_rl":
        mode = agentloop.get("observation_mode", "adaptive")
        if mode != "adaptive":
            raise ValueError(
                f"agentloop.observation_mode must be adaptive, got {mode!r}."
            )
        prompt = agentloop.get("prompt", "simple")
        if prompt not in (
            "simple",
            "simple_brief",
            "base",
            "base_gripper",
            "base_gripper_camera",
        ):
            raise ValueError(
                "agentloop.prompt must be simple, simple_brief, base, base_gripper or "
                f"base_gripper_camera, got {prompt!r}."
            )
        template_kwargs = agentloop.get("chat_template_kwargs") or {}
        if not isinstance(template_kwargs.get("enable_thinking", True), bool):
            raise ValueError(
                "agentloop.chat_template_kwargs.enable_thinking must be a boolean."
            )
        minimum_turns = 25 if cfg.runner.task_type == "reasoning" else 1
        if type(agentloop.max_turns) is not int or agentloop.max_turns < minimum_turns:
            raise ValueError(
                f"agentloop.max_turns must be an integer >= {minimum_turns} for "
                f"{cfg.runner.task_type}, got {agentloop.max_turns!r}."
            )
        compaction = agentloop.get("history_compaction")
        if compaction is not None and not (
            "target_tokens" in compaction or "max_messages" in compaction
        ):
            raise ValueError(
                "agentloop.history_compaction needs target_tokens (image-aware tiers) "
                "or max_messages (message window)."
            )
    if cfg.runner.task_type == "reasoning":
        if not cfg.critic.get("use_critic_model", False):
            raise ValueError(
                "RPent trains with PPO and needs critic.use_critic_model=True."
            )
        if not cfg.algorithm.recompute_logprobs and not cfg.rollout.get(
            "return_logprobs", False
        ):
            # Training scores from the engine's own behavior logprobs here.
            raise ValueError(
                "RPent without algorithm.recompute_logprobs needs "
                "rollout.return_logprobs=True."
            )
        if workflow == "libero_rl" and (
            cfg.actor.get("pack_traj", True)
            or not cfg.actor.get("expect_multi_modal_inputs", False)
        ):
            raise ValueError(
                "libero_rl requires actor.pack_traj=False and "
                "actor.expect_multi_modal_inputs=True for image rows."
            )
        _validate_training(cfg)
    return cfg


def _validate_training(cfg: DictConfig) -> None:
    algorithm = cfg.algorithm
    asynchronous = cfg.runner.get("async_mode", False)
    if asynchronous == cfg.runner.get("synchronous", False):
        raise ValueError(
            "RPent requires exactly one of runner.synchronous and runner.async_mode"
        )
    if (
        cfg.rollout.rollout_backend != "sglang"
        or cfg.rollout.sglang.backend_type != "server"
    ):
        raise ValueError("RPent requires the SGLang server backend")
    if algorithm.adv_type != ("sao_gae" if asynchronous else "gae_dynamic"):
        raise ValueError("RPent PPO requires gae_dynamic; async SAO requires sao_gae")
    if (
        algorithm.group_size != 1
        or algorithm.kl_beta != 0
        or algorithm.get("reinpp_kl_beta", 0) != 0
    ):
        raise ValueError("RPent PPO/SAO requires group_size=1 and zero KL coefficients")
    if (
        algorithm.get("loss_agg_func", "token-mean") != "token-mean"
        or not algorithm.use_valid_token_scale
        or algorithm.loss_scales
    ):
        raise ValueError(
            "RPent uses token-mean loss and uniform valid action-token scaling"
        )
    if cfg.agentloop.get("truncate_response", True):
        raise ValueError(
            "RPent requires agentloop.truncate_response=False to retain whole episodes"
        )
    from rlinf.algorithms.registry import get_critic_loss

    get_critic_loss(algorithm.critic_loss_type)
    if type(cfg.critic.update_epochs) is not int or cfg.critic.update_epochs < 1:
        raise ValueError("critic.update_epochs must be a positive integer")
    if cfg.critic.compute_advantages_after_update != asynchronous:
        raise ValueError(
            "SAO infers values after the critic update; synchronous PPO infers before it"
        )
    for role in ("actor", "critic"):
        worker = cfg[role]
        if (
            worker.training_backend != "megatron"
            or worker.model.pipeline_model_parallel_size != 1
            or worker.model.get("context_parallel_size", 1) != 1
        ):
            raise ValueError(
                "RPent PPO/SAO requires Megatron actor/critic with PP=CP=1"
            )
    correction = algorithm.get("importance_sampling_fix", False)
    if correction:
        clip = algorithm.get("importance_sampling_clip")
        if (
            not algorithm.recompute_logprobs
            or not cfg.rollout.return_logprobs
            or clip is None
            or not math.isfinite(clip)
            or clip <= 0
        ):
            raise ValueError(
                "TIS requires recomputed and behavior logprobs and a finite positive importance_sampling_clip"
            )
    if asynchronous:
        if (
            algorithm.loss_type != "dis"
            or algorithm.gae_lambda_alpha < 1
            or algorithm.n_minibatches != 1
        ):
            raise ValueError(
                "RPent SAO requires DIS, gae_lambda_alpha >= 1 and one minibatch"
            )
        if correction != algorithm.recompute_logprobs:
            raise ValueError(
                "Async recompute_logprobs and importance_sampling_fix must be enabled together"
            )
        if correction and cfg.agentloop.get("max_actor_accept_staleness") is None:
            raise ValueError(
                "Async TIS requires an explicit agentloop.max_actor_accept_staleness"
            )
        if cfg.rollout.pause_mode != "retract":
            raise ValueError(
                "RPent SAO requires retract pause to invalidate stale KV after weight updates"
            )
        validate_async_agent_cfg(
            cfg, RPentComponentPlacement(cfg, Cluster()), allow_recompute=True
        )
