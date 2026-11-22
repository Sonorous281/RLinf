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

"""Native rewards, cohort admission and replay of unfinished RPent episodes."""

import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from rlinf.agents.rpent.launcher import RPentEpisodeError
from rlinf.agents.rpent.rollout_buffer import (
    RPentCohortBuffer,
    RPentRolloutBufferWorker,
    fold_planner_turns,
)
from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.data.schema.reasoning_results import DynamicRolloutResult, GroupMetadata
from rlinf.scheduler import Worker


@pytest.mark.parametrize(
    "terminal,success,reward", [(True, False, 0), (True, True, 1), (False, False, None)]
)
def test_episode_reward_and_slot_release(terminal, success, reward):
    agent = object.__new__(RPentAgentLoopWorker)
    agent._env_slots = asyncio.Queue()
    agent._env_slots.put_nowait(3)
    agent._episode_launcher = SimpleNamespace(
        run=AsyncMock(return_value={"terminal": terminal, "success": success})
    )
    agent._planner_options = lambda url: {"base_url": url}
    agent.session_server_client = SimpleNamespace(
        session_base_url=lambda *_: "http://session/v1"
    )
    task = {"suite": "libero_goal", "task": 0, "seed": 4}
    query = agent.run_session_trajectory(
        SimpleNamespace(group_id=1, session_id="s"), [], [json.dumps(task)]
    )
    if reward is None:
        with pytest.raises(RPentEpisodeError):
            asyncio.run(query)
    else:
        assert asyncio.run(query) == (None, {}, reward)
    assert agent._env_slots.get_nowait() == 3
    assert agent._episode_launcher.run.await_args.args[1:] == (
        task,
        {"base_url": "http://session/v1"},
        3,
    )


def group(gid, version=2):
    return GroupMetadata(
        group_id=gid,
        task_name="rpent",
        earliest_version=version,
        completion_version=version,
        num_gen_tokens=8,
        num_generate_calls=2,
        rewards=[0, 1],
    )


@pytest.mark.parametrize("wrong_version", [False, True])
def test_cohort_admits_complete_groups_in_arrival_order(wrong_version):
    worker = object.__new__(RPentCohortBuffer)
    worker.cfg = OmegaConf.create(
        {
            "data": {"sampling_batch_size": 3, "rollout_batch_size": 2},
            "algorithm": {"group_size": 2},
        }
    )
    worker._next_request = AsyncMock(
        side_effect=[
            SimpleNamespace(group_ids=[gid], answers=[[str(gid)]]) for gid in range(3)
        ]
    )
    worker._prompts = Mock()
    worker._prompts.put.return_value.async_wait = AsyncMock()
    worker._incoming = Mock()
    worker._incoming.get.return_value.async_wait = AsyncMock(
        side_effect=[group(2), group(0, 1 if wrong_version else 2)]
    )
    if wrong_version:
        with pytest.raises(RuntimeError, match="spans policy versions"):
            asyncio.run(worker.release_batch(2))
    else:
        selected = asyncio.run(worker.release_batch(2))
        assert selected["selected"] == [2, 0]
        assert selected["remaining"] == [1]
        assert selected["rewards"] == [[0, 1], [0, 1]]


def test_cohort_sampler_consumes_all_candidates_before_starting_another_epoch(
    monkeypatch,
):
    monkeypatch.setattr(Worker, "__init__", lambda self: None)
    cfg = OmegaConf.create(
        {
            "runner": {"synchronous": True},
            "data": {
                "rollout_batch_size": 4,
                "sampling_batch_size": 5,
                "shuffle": False,
                "num_workers": 0,
            },
            "algorithm": {},
            "rollout_buffer": {
                "max_inflight_groups": 4,
                "max_buffered_groups": 8,
                "max_dropped_groups_per_step": 16,
            },
        }
    )
    worker = RPentCohortBuffer(cfg, range(10))
    assert list(worker.buffer.dataloader.batch_sampler) == [
        list(range(5)),
        list(range(5, 10)),
    ]
    assert worker.buffer.rollout_batch_size == 4


def test_checkpoint_replays_only_inflight_and_retains_unissued_requests(tmp_path):
    def make_worker():
        worker = object.__new__(RPentRolloutBufferWorker)
        worker.cfg = OmegaConf.create(
            {
                "algorithm": {"group_size": 1},
                "data": {"shuffle": False, "rollout_batch_size": 4},
            }
        )
        worker.task_name = "rpent"
        worker._fetch_lock = asyncio.Lock()
        loader = Mock(batch_size=4)
        loader.state_dict.return_value = {"cursor": 4}
        worker.buffer = SimpleNamespace(
            dataloader=loader,
            tasks={},
            group_tasks={},
            pending_release_version=None,
            next_group_id=0,
            num_samples_fetched=0,
            max_buffered_groups=4,
            max_inflight_groups=2,
        )
        return worker

    source = make_worker()
    source.buffer.tasks = {"rpent": {"buffer": {3: (group(3), "finished episode")}}}
    source.buffer.group_tasks = {4: "rpent"}
    source.buffer.next_group_id = 5
    source.buffer.num_samples_fetched = 6
    source._issued_requests = {4: "inflight episode"}
    source._queued_requests = deque([("next task", 5)])
    asyncio.run(source.save_checkpoint(str(tmp_path)))
    restored = make_worker()
    asyncio.run(restored.load_checkpoint(str(tmp_path), 2))
    assert restored.buffer.tasks == source.buffer.tasks
    assert list(restored._replay_requests) == ["inflight episode"]
    assert restored._queued_requests == source._queued_requests
    assert restored.buffer.next_group_id == 5
    restored.buffer.dataloader.load_state_dict.assert_called_once_with({"cursor": 4})
    restored.buffer.max_buffered_groups = 1
    with pytest.raises(ValueError, match="max_buffered_groups"):
        asyncio.run(restored.load_checkpoint(str(tmp_path), 2))


def test_folded_turns_exclude_tool_tokens_and_split_on_image_rewrite():
    result = DynamicRolloutResult(
        num_sequence=3,
        group_size=1,
        idx_to_traj=[0, 0, 0],
        input_ids=[[1, 2], [1, 2, 8, 3], [1, 2, 8, 3, 9, 4]],
        prompt_lengths=[1, 3, 5],
        response_lengths=[1, 1, 1],
        rollout_logprobs=[[-1.0], [-2.0], [-3.0]],
        rewards=[1.0] * 3,
        version_start=[2] * 3,
        version_end=[2] * 3,
        is_end=[True] * 3,
        extra_fields_turn={"rollout_weight_versions": [[2], [2], [2]]},
        multi_modal_inputs=[
            {"extends_previous_turn": flag} for flag in (False, True, False)
        ],
    )
    assert fold_planner_turns(result, 1) is result
    folded = fold_planner_turns(result, 3)
    assert folded.input_ids == result.input_ids[1:]
    assert folded.multi_modal_inputs == result.multi_modal_inputs[1:]
    batch = folded.to_actor_batch()
    assert batch["response_mask"][0].tolist() == [False, True, False, True]
    assert batch["rollout_logprobs"][0].tolist() == [0.0, -1.0, 0.0, -2.0]
    assert folded.extra_fields_turn["rollout_weight_versions"] == [[2, 2], [2]]
    assert folded.group_metadata().num_generate_calls == 3
    assert folded.group_metadata().num_gen_tokens == 3
    merged = DynamicRolloutResult.merge_result_list([folded, folded])
    restored = DynamicRolloutResult.split_results(merged, [2, 2])[1]
    assert restored.response_mask == folded.response_mask
    assert restored.extra_fields_turn == folded.extra_fields_turn


@pytest.mark.parametrize(
    "workflow,expected", [("episode", "auto"), ("libero_rl", "required")]
)
def test_batch_action_planner_keeps_grammar_out_of_training(
    workflow, expected, monkeypatch
):
    pytest.importorskip("sglang")
    from rlinf.agents.rpent.session_server import (
        RPentSessionServerWorker,
        SessionServerWorker,
    )

    chat = AsyncMock()
    monkeypatch.setattr(SessionServerWorker, "_chat_completions", chat)
    worker = object.__new__(RPentSessionServerWorker)
    worker._cfg = OmegaConf.create({"tools": {"rpent": {"workflow": workflow}}})
    body = {
        "tool_choice": "required",
        "temperature": 0.2,
        "tools": [{"type": "function"}],
    }
    asyncio.run(worker._chat_completions("episode-1", body))
    chat.assert_awaited_once_with(
        "episode-1", {"tool_choice": expected, "tools": body["tools"]}
    )
    assert body["tool_choice"] == "required"
