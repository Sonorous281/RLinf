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

"""RPent: append-only prompts, the episode-server protocol, retries and config."""

import asyncio
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _qwen35_chat import Qwen35ToyTokenizer, qwen35_agentic_tokenizer
from omegaconf import OmegaConf, open_dict

from rlinf.agents.rpent.episode import merge_prefix_turns
from rlinf.agents.rpent.history import (
    AppendOnlyHistory,
)
from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.workers.agent.agent_loop import AgentLoopOutput

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "look",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
]


@pytest.fixture
def qwen35(monkeypatch):
    return qwen35_agentic_tokenizer(monkeypatch)


def test_merge_prefix_turns_folds_image_rows_and_keeps_the_last_image_set():
    """Image turns merge on the token prefix; the row takes the last image set.

    Each turn's prompt expands every image of the chain in order, so a merged
    row carries the last merged turn's images, which match its merged tokens. A
    rewrite breaks the prefix and starts a new row with its own images.
    """
    import torch

    def images(count):
        return {
            "image_grid_thw": torch.ones(count, 3, dtype=torch.long),
            "num_images": count,
        }

    def turn(turn_id, prompt, response, count):
        return AgentLoopOutput(
            prompt_ids=prompt,
            response_ids=response,
            response_logprobs=[-float(token) for token in response],
            response_weight_versions=[0] * len(response),
            response_images=images(count),
            is_end=True,
            reward_score=1.0,
            extra_fields={"turn_id": turn_id, "tool_names": [f"tool{turn_id}"]},
        )

    turns = [
        turn(0, [1, 2], [3, 4], 1),  # image A
        # Adds a tool observation with image B; its prompt extends turn 0.
        turn(1, [1, 2, 3, 4, 9, 8], [5, 6], 2),
        # Compaction rewrote the history: a new row with its own image set.
        turn(2, [1, 2, 7], [8], 1),
    ]
    first, second = merge_prefix_turns(turns, max_scored_tokens=8)
    assert first.prompt_ids == [1, 2] and first.response_ids == [3, 4, 9, 8, 5, 6]
    assert first.response_mask == [1, 1, 0, 0, 1, 1]
    assert first.extra_fields["turn_ids"] == [0, 1]
    # The merged row holds every image of the chain, once each, in order.
    assert first.response_images["num_images"] == 2
    assert torch.equal(
        first.response_images["image_grid_thw"], torch.ones(2, 3, dtype=torch.long)
    )
    assert second.prompt_ids == [1, 2, 7] and second.response_images["num_images"] == 1


# Stands in for RPent's ``robots.libero_rl.rl_episode``: it plans against the
# worker's policy endpoint and writes the outcome, as the real process does.
FAKE_EPISODE = """
import argparse, json, os, urllib.error, urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--request", type=Path)
args, _ = parser.parse_known_args()
script = json.loads(os.environ["FAKE_RPENT"])
state = Path(script["state_dir"])
request = json.loads(args.request.read_text())
with (state / "calls.log").open("a") as calls:
    calls.write("started\\n")
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
planner = request["planner"]
messages, status = [{"role": "user", "content": "task"}], 200
for _ in range(script["turns"]):
    body = {"model": planner["model_name"], "messages": messages, "tools": script["tools"]}
    post = urllib.request.Request(
        planner["base_url"] + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(post, timeout=30) as response:
            message = json.loads(response.read())["choices"][0]["message"]
    except urllib.error.HTTPError as error:
        status = error.code
        break
    tool_call = message["tool_calls"][0]["id"]
    messages += [message, {"role": "tool", "tool_call_id": tool_call, "content": "seen"}]
error_kind = {200: None, 400: "context_limit"}.get(status, "backend")
outcome = {
    "success": error_kind is None,
    "truncated": False,
    "terminal": error_kind in (None, "context_limit"),
    "termination_reason": error_kind or "planner_completed",
    "error": "planner failed" if error_kind else None,
    "error_kind": error_kind,
    "init_state_id": 3,
    "memory_entries": [],
}
(args.output_dir / "outcome.json").write_text(json.dumps(outcome))
"""


class FakeRPent:
    """An episode process whose planner calls the policy endpoint ``turns`` times."""

    def __init__(self, *, turns=2):
        self.turns = turns
        self.state_dir: Path | None = None

    def lines(self, name: str) -> list[str]:
        path = self.state_dir / name
        return path.read_text().splitlines() if path.exists() else []

    @property
    def calls(self) -> list[str]:
        return self.lines("calls.log")


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    """Build an agent whose episode process and sampler are fakes."""
    from rlinf.scheduler import Cluster, Worker

    def init(worker):
        worker._rank, worker._world_size = 0, 1
        worker._group_name = "test-agents"
        worker._logger = logging.getLogger(__name__)
        worker._stacklevel = 1

    monkeypatch.setattr(Worker, "__init__", init)
    monkeypatch.setattr(
        Worker, "acquire_free_port", lambda self: Cluster.find_free_port()
    )
    monkeypatch.setattr(
        "rlinf.workers.agent.agent_loop.hf_tokenizer",
        lambda *a, **k: Qwen35ToyTokenizer(),
    )
    # Builds the patched agentic tokenizer class that skips sglang's markers.
    qwen35_agentic_tokenizer(monkeypatch)
    package = tmp_path / "fake-rpent" / "robots" / "libero_rl"
    package.mkdir(parents=True)
    for init_file in (package.parent / "__init__.py", package / "__init__.py"):
        init_file.touch()
    (package / "rl_episode.py").write_text(FAKE_EPISODE)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(package.parents[1]), os.environ.get("PYTHONPATH", "")]),
    )

    def make(
        fake,
        *,
        logprob=-0.5,
    ):
        fake.state_dir = tmp_path / f"fake-state-{len(list(tmp_path.iterdir()))}"
        fake.state_dir.mkdir()
        monkeypatch.setenv(
            "FAKE_RPENT",
            json.dumps(
                {
                    "turns": fake.turns,
                    "tools": TOOLS,
                    "state_dir": str(fake.state_dir),
                }
            ),
        )
        cfg = OmegaConf.create(
            {
                "runner": {
                    "task_type": "reasoning",
                    "seq_length": 4096,
                    "logger": {"log_path": str(tmp_path)},
                },
                "algorithm": {
                    "recompute_logprobs": True,
                    "sampling_params": {"max_new_tokens": 256},
                },
                "rollout": {
                    "return_logprobs": True,
                    "replica_affinity": True,
                    "model": {"model_path": "toy", "trust_remote_code": False},
                },
                "agentloop": {
                    "toolcall_parser": "qwen3.5",
                    "agentic_tokenizer": "qwen3_5",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "print_outputs": False,
                    "max_turns": 4,
                    "model_request_timeout_s": 5,
                    "episode_timeout_s": 5,
                    "minimum_output_tokens": 1,
                    "max_infra_retries": 1,
                    "max_infra_retry_rate": 0.05,
                    "merge_prefix_turns": True,
                },
                "tools": {
                    "rpent": {
                        "workflow": "libero_rl",
                        "python": sys.executable,
                        "sam3_endpoint": "http://sam3",
                        "libero_type": "pro",
                        "max_episode_steps": 10,
                        "env_gpus": [0, 1],
                        "episodes_per_worker": 2,
                        "output_dir": str(fake.state_dir / "episodes"),
                        "keep_episodes": "all",
                        "process_timeout_s": 30,
                    }
                },
            }
        )
        worker = RPentAgentLoopWorker(
            cfg, SimpleNamespace(rollout_num_model_instances=1)
        )
        tokenizer = worker.tokenizer
        worker.prompts = []

        async def generate(prompt_ids, sampling_params):
            turn = len(worker.prompts)
            response = tokenizer(
                f"step {turn}\n\n<tool_call>\n<function=look>\n"
                "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call><|im_end|>"
            )["input_ids"]
            worker.prompts.append((prompt_ids, sampling_params, response))
            return {
                "output_ids": response,
                "prompt_tokens": len(prompt_ids),
                "logprobs": [logprob] * len(response),
                "finish_reason": "stop",
            }

        worker.generate = generate
        return worker

    return make


def run_query(worker):
    async def run():
        await worker._model_server.start()
        try:
            return await worker.run_one_query([], [json.dumps({"suite": "goal"})])
        finally:
            await worker._model_server.stop()

    return asyncio.run(asyncio.wait_for(run(), timeout=60))


def test_episode_turns_merge_into_one_masked_row_scored_by_the_outcome(make_agent):
    fake = FakeRPent()
    worker = make_agent(fake)
    result = run_query(worker)
    (row,) = result.single_turn_outputs
    (first, _, first_response), (second, _, second_response) = worker.prompts
    gap = len(second) - len(first) - len(first_response)
    # The planner's echoed reply continues the sampled tokens.
    assert second[: len(first) + len(first_response)] == first + first_response
    assert row.prompt_ids == first
    assert row.prompt_ids + row.response_ids == second + second_response
    # Only sampled tokens are scored; the tool result between turns is not.
    assert row.response_mask == (
        [1] * len(first_response) + [0] * gap + [1] * len(second_response)
    )
    assert row.response_logprobs == [-0.5 if m else 0.0 for m in row.response_mask]
    assert row.reward_score == 1.0 and row.extra_fields["turn_ids"] == [0, 1]
    assert (
        result.extra_fields["turns"] == 2 and result.extra_fields["init_state_id"] == 3
    )
    assert result.extra_fields["success"] and result.extra_fields["infra_retries"] == 0
    assert len(fake.calls) == 1


@pytest.mark.parametrize("logprob", [math.nan, math.inf])
def test_non_finite_behavior_logprobs_fail_without_retry(make_agent, logprob):
    fake = FakeRPent(turns=1)
    with pytest.raises(ValueError, match="finite behavior logprob"):
        run_query(make_agent(fake, logprob=logprob))
    assert len(fake.calls) == 1


# Shared synchronous agent PPO settings are covered by the SearchR1 tests.


def test_token_space_history_preserves_sampled_ids_and_expands_only_new_images(qwen35):
    import base64
    import io

    from PIL import Image

    from rlinf.agents.rpent.history import TemplateMessageError

    def image_part(color):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
        return {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode()
            },
        }

    history = AppendOnlyHistory(qwen35)
    first_request = {"messages": [{"role": "user", "content": [image_part("red")]}]}
    compact = history.render_token_space(first_request)
    start, end = compact.image_spans[0]
    expanded = (
        compact.input_ids[:start]
        + compact.input_ids[start:end] * 4
        + compact.input_ids[end:]
    )
    history.accept_token_space(
        {
            "prompt_token_ids": expanded,
            "image_spans": [(start, start + 4)],
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
    )
    response = qwen35.tokenizer("look again<|im_end|>")["input_ids"]
    assistant = {"role": "assistant", "content": "look again"}
    history.record(first_request, expanded, response, assistant)
    next_request = {
        "messages": [
            *first_request["messages"],
            assistant,
            {"role": "user", "content": [image_part("blue")]},
        ]
    }
    following = history.render_token_space(next_request)
    assert following.input_ids[: len(expanded + response)] == expanded + response
    assert following.prefix_length == len(expanded + response)
    assert following.image_spans[0] == (start, start + 4)
    assert following.image_spans[1][1] - following.image_spans[1][0] == 1
    assert following.pixel_values is None
    # Compaction starts a new chain; previous image occurrences cannot leak in.
    reset = history.render_token_space(
        {"messages": [{"role": "user", "content": [image_part("blue")]}]}
    )
    assert reset.prefix_length == 0 and len(reset.images) == 1
    with pytest.raises(TemplateMessageError, match="Literal vision"):
        history.render_token_space(
            {"messages": [{"role": "user", "content": "print <|image_pad|>"}]}
        )


@pytest.mark.parametrize(
    "key", ["enable_shared_multi_modal_processor", "enable_token_space_processor"]
)
def test_token_space_checks_exact_served_prompt_ids(make_agent, key):
    worker = make_agent(FakeRPent(turns=1))
    worker.cfg.rollout.sglang = {key: True}
    generated = {
        "output_ids": [9],
        "prompt_tokens": 2,
        "prompt_token_ids": [1, 3],
        "logprobs": [-0.2],
        "weight_versions": [0],
    }
    with pytest.raises(ValueError, match="different prompt IDs"):
        worker._check_generation(generated, [1, 2])
    generated["prompt_token_ids"] = [1, 2]
    worker._check_generation(generated, [1, 2])
    generated["output_ids"] = [worker.tokenizer.convert_tokens_to_ids("<|image_pad|>")]
    with pytest.raises(ValueError, match="Generated vision markers"):
        worker._check_generation(generated, [1, 2])
    del generated["prompt_token_ids"]
    with pytest.raises(ValueError, match="different prompt IDs"):
        worker._check_generation(generated, [1, 2])


def compose(overrides=(), config_name="train_qwen3.5_4b_libero_rl_ppo"):
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).parents[2] / "examples/agent/rpent/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    # Templates require deployment paths; CPU tests supply explicit stand-ins.
    for name, value in {
        "runner.output_dir": "/test/training",
        "data.train_data_paths": ["/test/tasks.jsonl"],
        "data.val_data_paths": ["/test/eval.jsonl"],
        "tools.rpent.python": "/test/env/bin/python",
        "tools.rpent.libero_config_path": "/test/libero-config",
        "tools.rpent.output_dir": "/test/episodes",
        "rollout.model.model_path": "/test/full-model",
    }.items():
        parent, key = name.rsplit(".", 1)
        node = OmegaConf.select(cfg, parent)
        if node is not None and OmegaConf.is_missing(node, key):
            OmegaConf.update(cfg, name, value)
    return cfg


@pytest.mark.parametrize(
    "config_name",
    [
        "train_qwen3.5_4b_libero_rl_ppo",
        "eval_qwen3.5_4b_libero_rl",
        "train_qwen3.5_4b_libero_rl_sao",
        "train_qwen3.5_4b_libero_rl_sao_local",
        "train_qwen3.5_4b_libero_rl_sao_3node",
    ],
)
def test_example_configs_preserve_planner_and_training_contracts(config_name):
    """Training ratios use full support; evaluation keeps planner sampling."""
    cfg = compose(config_name=config_name)
    sampling = cfg.algorithm.sampling_params
    if config_name.startswith("train_"):
        assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 1.0, -1)
        assert sampling.presence_penalty == 0.0
        assert sampling.repetition_penalty == 1.0
        assert cfg.agentloop.thinking_token_budget is None
        assert cfg.rollout.sglang.backend_type == "server"
    else:
        assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 0.95, 20)
    assert sampling.max_new_tokens == 32768
    assert cfg.agentloop.max_turns == 100
    assert cfg.agentloop.observation_mode == "adaptive"
    placement = cfg.cluster.component_placement
    group = cfg.cluster.node_groups[0]
    assert (
        placement.get("rollout", placement.get("rollout,reward")).node_group
        == group.label
    )
    assert {
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "rlinf.hybrid_engines.sglang.multimodal"
    } in group.env_configs[0].env_vars
    if cfg.runner.task_type == "reasoning":
        for component in (cfg.actor, cfg.critic):
            assert not component.megatron.ckpt_convertor.hf_model_path.endswith("-text")
            assert component.model.freeze_vision_model
            assert component.model.freeze_vision_projection
    assert cfg.agentloop.chat_template_kwargs.enable_thinking is True
    assert cfg.runner.seq_length == 262144
    assert cfg.tools.rpent.process_timeout_s > cfg.agentloop.episode_timeout_s
    compaction = cfg.agentloop.history_compaction
    if "target_tokens" in compaction:
        assert compaction.target_tokens <= 0.7 * cfg.runner.seq_length
        # A compacted prompt still gets the whole response plus schema headroom.
        assert cfg.runner.seq_length - compaction.target_tokens >= 32768 + 16384
    else:
        assert compaction.context_window == cfg.runner.seq_length
        assert compaction.max_fraction == 0.7


@pytest.mark.parametrize("field", ["cache_bytes", "prefetch_batches", "prefetch_bytes"])
def test_legacy_image_budgets_normalize_and_reject_conflicts(field):
    from rlinf.config import validate_multimodal_learner_cfg, validate_rpent_cfg

    cfg = compose(
        [
            "rollout.sglang.enable_shared_multi_modal_processor=true",
            "cluster.node_groups.0.env_configs.0.env_vars.0.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=''",
            "tools.rpent.store_images_on_disk=true",
        ]
    )
    budgets = {"cache_bytes": 1024, "prefetch_batches": 2, "prefetch_bytes": 4096}
    for key, value in budgets.items():
        OmegaConf.update(
            cfg, f"tools.rpent.training_image_{key}", value, force_add=True
        )
    OmegaConf.set_struct(cfg, True)
    validate_rpent_cfg(cfg)
    validate_multimodal_learner_cfg(cfg)
    assert dict(cfg.actor.multimodal) == dict(cfg.critic.multimodal) == budgets
    snapshot = OmegaConf.to_container(cfg)
    validate_rpent_cfg(cfg)
    assert OmegaConf.to_container(cfg) == snapshot
    OmegaConf.update(cfg, f"critic.multimodal.{field}", budgets[field] + 1)
    with pytest.raises(ValueError, match=f"critic.multimodal.{field} conflicts"):
        validate_rpent_cfg(cfg)
    del cfg.tools.rpent
    # Generic consumers can choose independent actor and critic budgets.
    validate_multimodal_learner_cfg(cfg)
    assert cfg.actor.multimodal[field] == budgets[field]
    assert cfg.critic.multimodal[field] == budgets[field] + 1


def test_forced_token_has_zero_loss_gradient_and_preserves_normalization():
    from rlinf.agents.rpent.thinking_budget import ThinkingBudget

    budget = ThinkingBudget(2, 5, 10)
    ids = torch.tensor([1, 2, 5, 3])
    mask = torch.tensor(budget.response_mask(ids.tolist(), [2]))
    logits = torch.arange(32, dtype=torch.float64).reshape(4, 8).requires_grad_()
    logprobs = logits.log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
    loss = -(logprobs * mask).sum() / mask.sum()
    reference = -logprobs[[0, 1, 3]].mean()
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(grad, reference_grad)
    assert torch.count_nonzero(grad[2]).item() == 0
    assert all(torch.count_nonzero(grad[row]).item() for row in [0, 1, 3])


@pytest.mark.parametrize(
    "config_name",
    [
        "eval_qwen3.5_4b_libero_rl_smoke_a100",
        "eval_qwen3.5_4b",
        "train_qwen3.5_4b_ppo",
        "train_qwen3.5_4b_libero_rl_ppo",
        "train_qwen3.5_4b_libero_rl_sao_a100",
        "train_qwen3.5_4b_libero_rl_sao_a100_8gpu",
    ],
)
def test_rpent_example_validation_and_component_placement(
    config_name, monkeypatch, tmp_path
):
    from test_placement import create_fake_cluster

    from rlinf.agents.rpent import config_validation
    from rlinf.agents.rpent.placement import RPentComponentPlacement
    from rlinf.config import (
        validate_multimodal_learner_cfg,
        validate_qwen_vl_chunked_cfg,
    )
    from rlinf.utils.placement import ModelParallelComponentPlacement

    cfg = compose(config_name=config_name)
    cluster = create_fake_cluster(
        num_nodes=cfg.cluster.num_nodes,
        accelerators_per_node=8,
        extra_group_mapping={
            group.label: list(range(cfg.cluster.num_nodes))
            for group in cfg.cluster.get("node_groups", [])
        },
    )
    import rlinf.config as common_config

    cfg.runner.output_dir = str(tmp_path)
    monkeypatch.setattr(common_config, "Cluster", lambda *a, **kw: cluster)
    monkeypatch.setattr(config_validation, "Cluster", lambda: cluster)
    if cfg.runner.task_type == "reasoning_eval":
        common_config.validate_cfg(cfg)
    else:
        # Megatron-specific model validation needs the training dependencies.
        common_config.validate_reasoning_cfg(cfg)
        config_validation.validate_rpent_cfg(cfg)
    assert cfg.agentloop.get("thinking_token_budget") is None
    if cfg.runner.task_type != "reasoning":
        assert "actor" not in cfg and "critic" not in cfg
        with open_dict(cfg.agentloop):
            cfg.agentloop.thinking_token_budget = -1
        with pytest.raises(ValueError, match="nonnegative integer"):
            common_config.validate_cfg(cfg)
        return
    validate_multimodal_learner_cfg(cfg)
    validate_qwen_vl_chunked_cfg(cfg)
    placement = RPentComponentPlacement(cfg, cluster)
    assert not placement.is_pipeline
    if cfg.runner.get("async_mode", False):
        assert placement.is_disaggregated
        assert cfg.algorithm.critic_loss_type == "mse"
        assert cfg.critic.compute_advantages_after_update
        # Only the RPent adapter accepts this shared-learner layout.
        with pytest.raises(ValueError, match="placement"):
            ModelParallelComponentPlacement(cfg, cluster)
        cfg.algorithm.importance_sampling_clip = float("inf")
        with pytest.raises(ValueError, match="finite positive"):
            config_validation.validate_rpent_cfg(cfg)
