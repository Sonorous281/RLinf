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

"""Train an RPent planner with synchronous PPO or asynchronous SAO PPO."""

import json

import hydra
from omegaconf import OmegaConf

from rlinf.agents.rpent.placement import RPentComponentPlacement
from rlinf.agents.rpent.rollout_buffer import RPentRolloutBufferWorker
from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import output_redirector
from rlinf.workers.rollout.utils import get_rollout_backend_worker


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    from rlinf.agents.rpent.async_runner import RPentAsyncRunner
    from rlinf.runners.agent_ppo_runner import AgentPPORunner
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

    cfg = validate_cfg(cfg)
    get_logger().info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = RPentComponentPlacement(cfg, cluster)

    rollout_strategy = placement.get_strategy("rollout")
    rollout = (
        get_rollout_backend_worker(cfg)
        .create_group(cfg, placement)
        .launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_strategy,
        )
    )
    agent_loop = RPentAgentLoopWorker.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=placement.get_strategy("agentloop"),
    )
    actor = MAMegatronActor.create_group(cfg, placement).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    critic = MAMegatronCritic.create_group(cfg, placement).launch(
        cluster,
        name=cfg.critic.group_name,
        placement_strategy=placement.get_strategy("critic"),
    )

    tokenizer = hf_tokenizer(cfg.actor.tokenizer.tokenizer_model)
    train_ds, val_ds = create_reasoning_datasets(cfg, tokenizer)
    runner_cls = AgentPPORunner if cfg.runner.synchronous else RPentAsyncRunner
    runner_kwargs = {}
    if cfg.runner.get("async_mode", False):
        runner_kwargs["rollout_buffer"] = RPentRolloutBufferWorker.create_group(
            cfg, train_ds
        ).launch(
            cluster, name="RolloutBuffer", placement_strategy=NodePlacementStrategy([0])
        )
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
        **runner_kwargs,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
