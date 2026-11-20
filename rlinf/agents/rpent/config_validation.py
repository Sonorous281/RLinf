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


"""Configuration contracts for RPent planner workflows."""

from omegaconf import DictConfig, OmegaConf, open_dict


def validate_rpent_cfg(cfg: DictConfig) -> DictConfig:
    """Validate RPent resources, planner inputs and the selected RL algorithm."""
    rpent = cfg.get("tools", {}).get("rpent")
    if rpent is None:
        return cfg
    _validate_resources(rpent)
    _validate_multimodal(cfg, rpent)
    _validate_planner(cfg, rpent)
    _validate_training(cfg, rpent)
    return cfg


def _validate_resources(rpent: DictConfig) -> None:
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
    if type(rpent.get("record_api_trace", False)) is not bool:
        raise ValueError("tools.rpent.record_api_trace must be a boolean.")


def _validate_multimodal(cfg: DictConfig, rpent: DictConfig) -> None:
    from rlinf.config import (
        release_image_replay_enabled,
        shared_multi_modal_processor_enabled,
    )

    if shared_multi_modal_processor_enabled(OmegaConf.select(cfg, "rollout.sglang")):
        if cfg.rollout.rollout_backend != "sglang" or not rpent.get(
            "store_images_on_disk", False
        ):
            raise ValueError(
                "Shared multimodal preprocessing for RPent requires SGLang and store_images_on_disk=True"
            )
        if release_image_replay_enabled(cfg.rollout):
            if cfg.rollout.sglang.get("enable_token_space_preprocess_cache", False):
                raise ValueError(
                    "Release image replay stores captured pixels; disable the native TokenSpace cache"
                )
            if cfg.agentloop.get("thinking_token_budget") is not None:
                raise ValueError(
                    "Release image replay requires thinking_token_budget=null"
                )
        elif cfg.rollout.get("env_vars", {}).get(
            "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"
        ):
            raise ValueError(
                "Remove SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE when enabling shared multimodal preprocessing"
            )
        learner_roles = (
            ("actor", "critic") if cfg.runner.task_type == "reasoning" else ()
        )
        for role in learner_roles:
            worker = cfg[role]
            if role == "critic" and not worker.use_critic_model:
                continue
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


def _validate_planner(cfg: DictConfig, rpent: DictConfig) -> None:
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
    world_cord = rpent.get("world_cord", "base")
    if world_cord not in {"base", "general"}:
        raise ValueError("tools.rpent.world_cord must be base or general")
    if world_cord != "base" and workflow not in {"episode", "pear"}:
        raise ValueError(
            "general coordinates require the adaptive episode or pear workflow"
        )
    if workflow == "pear" and rpent.get("action_mode") != "single":
        raise ValueError("pear requires action_mode=single")
    if workflow not in ("libero", "libero_rl", "episode", "pear"):
        raise ValueError(
            f"tools.rpent.workflow must be libero, libero_rl, episode or pear, got {workflow!r}."
        )
    if workflow in {"libero_rl", "episode", "pear"}:
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
        if workflow in {"episode", "pear"}:
            if rpent.get("entrypoint") != "robots.rl.rl_episode":
                raise ValueError(
                    "episode requires tools.rpent.entrypoint=robots.rl.rl_episode."
                )
            if rpent.get("action_mode") not in {"single", "batch"}:
                raise ValueError("episode action_mode must be single or batch.")
            if (
                type(rpent.get("max_actions_per_call")) is not int
                or rpent.max_actions_per_call < 1
            ):
                raise ValueError(
                    "episode max_actions_per_call must be a positive integer."
                )
            if rpent.get("perception_mode") != "single":
                raise ValueError("remote episode PPO requires single perception.")
            if rpent.get("image_size") not in {256, 512, 1024}:
                raise ValueError("episode image_size must be 256, 512 or 1024.")
            for name in ("reasoning_history", "image_history"):
                if type(rpent.get(name)) is not int or rpent[name] < -1:
                    raise ValueError(f"episode {name} must be an integer >= -1.")
            if agentloop.minimum_output_tokens != max_new_tokens:
                raise ValueError(
                    "episode must reserve the full configured max_new_tokens."
                )
            if compaction is not None and cfg.algorithm.adv_type == "gae_dynamic":
                raise ValueError(
                    "compressed episode histories require chronological actor_gae, not gae_dynamic."
                )


def _validate_training(cfg: DictConfig, rpent: DictConfig) -> None:
    from rlinf.config import validate_agent_ppo_cfg, validate_async_agent_ppo_cfg

    workflow = rpent.get("workflow", "libero")
    if cfg.runner.task_type == "reasoning":
        if not cfg.critic.get("use_critic_model", False):
            if (
                not cfg.runner.get("synchronous", False)
                or cfg.algorithm.adv_type != "grpo_dynamic"
                or cfg.algorithm.group_size <= 1
                or cfg.algorithm.advantage_mode != "trajectory"
                or cfg.algorithm.normalize_advantages
            ):
                raise ValueError(
                    "RPent without a critic requires synchronous grpo_dynamic, "
                    "group_size > 1, advantage_mode=trajectory and "
                    "normalize_advantages=False (GRPO already normalizes each group)."
                )
            if (
                cfg.actor.training_backend != "megatron"
                or cfg.rollout.rollout_backend != "sglang"
                or cfg.actor.model.pipeline_model_parallel_size != 1
                or cfg.actor.model.get("context_parallel_size", 1) != 1
                or cfg.algorithm.kl_beta != 0
                or cfg.algorithm.get("reinpp_kl_beta", 0) != 0
                or cfg.algorithm.loss_agg_func != "token-mean"
                or not cfg.algorithm.use_valid_token_scale
            ):
                raise ValueError(
                    "RPent GRPO requires Megatron PP=CP=1, SGLang, zero KL, "
                    "token-mean and valid-action-token normalization."
                )
        if not cfg.algorithm.recompute_logprobs and not cfg.rollout.get(
            "return_logprobs", False
        ):
            # Training scores from the engine's own behavior logprobs here.
            raise ValueError(
                "RPent without algorithm.recompute_logprobs needs "
                "rollout.return_logprobs=True."
            )
        if workflow in {"libero_rl", "episode", "pear"} and (
            cfg.actor.get("pack_traj", True)
            or not cfg.actor.get("expect_multi_modal_inputs", False)
        ):
            raise ValueError(
                "adaptive RPent requires actor.pack_traj=False and "
                "actor.expect_multi_modal_inputs=True for image rows."
            )
        if cfg.runner.get("synchronous", False) and cfg.critic.use_critic_model:
            validate_agent_ppo_cfg(cfg, "RPent")
        elif not cfg.runner.get("synchronous", False):
            validate_async_agent_ppo_cfg(cfg, "RPent")
