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

"""Train an RPent planner with synchronous PPO/GRPO or asynchronous SAO PPO."""

import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.runners.agent_ppo_runner import AgentPPORunner
from rlinf.runners.async_agent_ppo_runner import AsyncAgentPPORunner
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.rollout.utils import get_rollout_backend_worker


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    cfg = validate_cfg(cfg)
    get_logger().info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    agent_loop_cls = RPentAgentLoopWorker
    if cfg.get("replay") and cfg.runner.resume_dir is None:
        if not cfg.runner.synchronous or cfg.replay.expected_version != 0:
            raise ValueError(
                "Saved-cohort recovery requires a fresh synchronous policy"
            )
        from rlinf.agents.rpent.replay import (
            RPentRecoveryAgentLoopWorker,
            SavedRPentBatch,
        )

        saved_batch = SavedRPentBatch(
            Path(cfg.replay.source_dir),
            Path(cfg.replay.manifest),
            cfg.replay.expected_version,
        )
        saved_batch.verify_policy(Path(cfg.rollout.model.model_path))
        agent_loop_cls = RPentRecoveryAgentLoopWorker

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelComponentPlacement(cfg, cluster)

    rollout_strategy = placement.get_strategy("rollout")
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement)
        .launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_strategy,
            env_vars=OmegaConf.to_container(
                cfg.rollout.get("env_vars", {}), resolve=True
            )
            if cfg.rollout.get("env_vars")
            else None,
        )
    )
    agent_loop = agent_loop_cls.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=placement.get_strategy("agentloop"),
    )
    actor = MAMegatronActor.create_group(cfg, placement).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
        env_vars=OmegaConf.to_container(cfg.actor.get("env_vars", {}), resolve=True)
        if cfg.actor.get("env_vars")
        else None,
    )
    critic = None
    if cfg.critic.use_critic_model:
        from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

        critic = MAMegatronCritic.create_group(cfg, placement).launch(
            cluster,
            name=cfg.critic.group_name,
            placement_strategy=placement.get_strategy("critic"),
            env_vars=OmegaConf.to_container(
                cfg.critic.get("env_vars", {}), resolve=True
            )
            if cfg.critic.get("env_vars")
            else None,
        )

    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_reasoning_datasets(cfg, tokenizer)
    runner_cls = AgentPPORunner if cfg.runner.synchronous else AsyncAgentPPORunner
    runner = runner_cls(
        cfg=cfg,
        placement=placement,
        train_dataset=train_ds,
        val_dataset=val_ds,
        rollout=rollout,
        inference=None,
        actor=actor,
        reward=None,
        agent_loop=agent_loop,
        critic=critic,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
