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

"""Channel-backed completion ordering, critic targets, and policy staleness."""

import asyncio
import copy
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.schema.agent_trajectory import (
    BufferedTrajectory,
    construct_rollout_batch,
)
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel
from rlinf.scheduler.channel.channel_worker import LocalChannel


def channel(capacity=0):
    result = Channel()
    result._initialize(
        "async-completions", None, None, None, local_channel=LocalChannel(capacity)
    )
    return result


class AsyncQueueChannel:
    """Async transport double; LocalChannel only supports synchronous calls."""

    def __init__(self, capacity=0):
        self.queue = asyncio.Queue(capacity)

    def put(self, item, async_op=False):
        if async_op:
            return SimpleNamespace(async_wait=lambda: self.queue.put(item))
        self.queue.put_nowait(item)

    def get(self, *, async_op):
        assert async_op
        return SimpleNamespace(async_wait=self.queue.get)

    def get_nowait(self):
        return self.queue.get_nowait()

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()


def trajectory(sample_id=0, versions=None):
    if versions is None:
        versions = [[0, 1], [2]]
    result = DynamicRolloutResult(
        num_sequence=2,
        group_size=1,
        idx_to_traj=[0, 0],
        input_ids=[[1, 2, 3], [1, 2, 3, 4, 5]],
        prompt_lengths=[1, 4],
        response_lengths=[2, 1],
        rollout_logprobs=[[-0.2, -0.3], [-0.4]],
        rollout_weight_versions=versions,
        rollout_ids=[sample_id, sample_id],
        is_end=[False, True],
        rewards=[1.0, 1.0],
    )
    return BufferedTrajectory(sample_id, [1], "answer", result, {"__sum__/turns": 2})


def consume(input_channel, output_channel, retry_channel, **kwargs):
    return construct_rollout_batch(
        input_channel=input_channel,
        output_channel=output_channel,
        retry_channel=retry_channel,
        batch_size=kwargs.pop("batch_size", 1),
        seq_length=6,
        gamma=0.5,
        published_version=kwargs.pop("published_version", 2),
        **kwargs,
    )


@pytest.mark.parametrize(
    "staleness,stale", [(None, False), (0, True), (2, True), (3, False)]
)
def test_staleness_is_checked_at_consumption_and_retries_keep_the_prompt(
    staleness, stale
):
    incoming, output, retry = channel(2), channel(), channel()
    incoming.put(trajectory(9, versions=[[4, 5], [6]]))
    incoming.put(trajectory(9, versions=[[7, 7], [7]]))
    state = consume(
        incoming, output, retry, published_version=7, max_weight_staleness=staleness
    )
    assert state["sample_ids"] == [9]
    assert state["stale_dropped"] == int(stale)
    assert state["oldest_version"] == (7 if stale else 4)
    assert output.qsize() == 1
    if stale:
        assert retry.get_nowait() == (9, [1], "answer")
        assert incoming.empty()
    else:
        assert retry.empty() and incoming.qsize() == 1


def test_checkpoint_snapshot_preserves_fifo_and_behavior_probabilities():
    incoming, restored, output, retry = channel(2), channel(2), channel(), channel()
    for sample_id in (10, 5):
        incoming.put(trajectory(sample_id))
    snapshot = copy.deepcopy(incoming.peek_all())
    assert incoming.qsize() == 2
    for entry in snapshot:
        restored.put(entry)
    for sample_id in (10, 5):
        state = consume(restored, output, retry, published_version=8)
        assert state["sample_ids"] == [sample_id]
        result = output.get_nowait()
        assert result.rollout_logprobs == [[-0.2, -0.3], [-0.4]]
        assert result.rollout_weight_versions == [[0, 1], [2]]
    assert restored.empty() and incoming.qsize() == 2


@pytest.mark.parametrize(
    "versions,published_version,error",
    [
        ([[1], [2]], 8, "Each generated token"),
        ([[1, 2], None], 8, "Each generated token"),
        ([[2, 2], [2]], 1, "newer than the published version"),
    ],
)
def test_consumption_rejects_invalid_behavior_provenance(
    versions, published_version, error
):
    incoming, output, retry = channel(1), channel(), channel()
    incoming.put(trajectory(versions=versions))
    with pytest.raises(ValueError, match=error):
        consume(incoming, output, retry, published_version=published_version)
    assert incoming.empty() and output.empty() and retry.empty()


@pytest.fixture
def producer_worker(monkeypatch):
    from omegaconf import OmegaConf

    from rlinf.workers.agent.agent_loop import AgentLoopWorker, MultiAgentLoopWorker

    def init_hardware(self, cfg, placement):
        self.cfg = cfg
        self._rank = 0
        self._world_size = 1

    monkeypatch.setattr(AgentLoopWorker, "__init__", init_hardware)
    return lambda: MultiAgentLoopWorker(
        OmegaConf.create(
            {
                "runner": {"task_type": "reasoning"},
                "rollout": {"max_in_flight_rollouts": 2},
                "data": {"rollout_batch_size": 2},
            }
        ),
        None,
    )


def test_byte_backpressure_preserves_a_complete_batch_and_releases_consumed_results(
    producer_worker,
):
    async def run():
        worker = producer_worker()
        worker.cfg.rollout.max_buffer_bytes = 1
        incoming, outgoing = AsyncQueueChannel(), AsyncQueueChannel(8)
        started = []

        async def rollout(*, input_dict, **kwargs):
            started.append(input_dict[0])
            return trajectory(input_dict[0]).result, {}

        worker.run_agentloop_rollout_group = rollout
        for index in range(4):
            incoming.put((index, [index], "answer"))
        producer = asyncio.create_task(worker.run_async_rollout(incoming, outgoing))
        try:
            while outgoing.qsize() < 2:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert started == [0, 1]
            assert worker.get_async_buffer_bytes()["memory_admission_paused"] == 1
            consumed = [outgoing.get_nowait().sample_id for _ in range(2)]
            assert consumed == [0, 1]
            # The learner can drain safely while queued admissions wait for bytes.
            await worker.pause_rollouts()
            worker.release_rollout_bytes(consumed)
            assert worker.get_async_buffer_bytes()["retained_bytes_estimate"] == 0
            await asyncio.sleep(0)
            assert started == [0, 1]
            await worker.resume_rollouts()
            while outgoing.qsize() < 2:
                await asyncio.sleep(0)
            assert [outgoing.get_nowait().sample_id for _ in range(2)] == [2, 3]
        finally:
            await worker.stop_rollouts()
            await producer

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_producer_drain_waits_for_a_blocked_result_send(producer_worker):
    async def run():
        worker = producer_worker()
        incoming, outgoing = AsyncQueueChannel(), AsyncQueueChannel(1)
        outgoing.put(trajectory(9))
        incoming.put((0, [0], "answer"))

        async def rollout(**kwargs):
            return trajectory(0).result, {}

        worker.run_agentloop_rollout_group = rollout
        producer = asyncio.create_task(worker.run_async_rollout(incoming, outgoing))
        try:
            while not worker.get_async_rollout_state()["waiting_for_buffer"]:
                await asyncio.sleep(0)
            draining = asyncio.create_task(worker.pause_rollouts())
            await asyncio.sleep(0)
            assert not draining.done()
            assert outgoing.get_nowait().sample_id == 9
            assert await draining == {"in_flight": 0, "waiting_for_buffer": 0}
            assert outgoing.get_nowait().sample_id == 0
            assert worker.get_async_rollout_state() == {
                "in_flight": 0,
                "waiting_for_buffer": 0,
            }
        finally:
            if not producer.done():
                await worker.stop_rollouts()
            await producer

    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.fixture
def async_data_runner():
    from omegaconf import OmegaConf
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    from rlinf.data.datasets.reasoning import collate_fn
    from rlinf.runners.async_agent_ppo_runner import AsyncAgentPPORunner

    items = [
        SimpleNamespace(
            prompt=[i],
            length=1,
            idx=i,
            answer=[str(i)],
            solution=None,
            image_data=None,
            prompt_text=None,
            meta=None,
            multi_modal_inputs=None,
        )
        for i in range(32)
    ]

    def create(shuffle):
        runner = object.__new__(AsyncAgentPPORunner)
        runner.cfg = OmegaConf.create(
            {
                "runner": {"max_epochs": 10},
                "data": {"rollout_batch_size": 8, "shuffle": shuffle},
                "rollout": {"rollout_buffer_capacity": 8},
            }
        )
        runner.max_steps = 40
        generator = torch.Generator().manual_seed(1234)
        sampler = (
            RandomSampler(items, generator=generator)
            if shuffle
            else SequentialSampler(items)
        )
        runner.train_dataloader = StatefulDataLoader(
            items,
            batch_size=8,
            drop_last=True,
            num_workers=0,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        runner._next_sample_id = runner._producer_epoch = runner._stale_count = 0
        runner._prompt_queue = deque()
        runner._data_iterator = None
        runner._pending = {}
        runner.async_rollout_channel = channel()
        runner.published_version = runner.actor_version = runner.actor_updates = 1
        runner.critic_updates = 2
        return runner

    return create


@pytest.mark.parametrize(
    "shuffle,old_key,new_key",
    [
        (True, None, None),
        (False, None, None),
        (True, "enable_token_space_processor", "enable_shared_multi_modal_processor"),
        (True, "enable_shared_multi_modal_processor", "enable_token_space_processor"),
    ],
)
def test_async_resume_preserves_producer_sampler_order(
    async_data_runner, shuffle, old_key, new_key
):
    reference = async_data_runner(shuffle)
    if old_key:
        reference.cfg.rollout.sglang = {old_key: True}
    reference.global_steps = 3
    for _ in range(40):
        prompt = reference._next_prompt()
    reference._pending = {prompt[0]: prompt}
    snapshot = copy.deepcopy(reference._save_data_state())
    assert snapshot["async_rollout"]["token_space"] is (old_key is not None)
    expected = [reference._next_prompt() for _ in range(48)]

    restored = async_data_runner(shuffle)
    if new_key:
        restored.cfg.rollout.sglang = {new_key: True}
    restored.global_steps = reference.global_steps
    restored._load_data_state(snapshot)

    assert [restored._next_prompt() for _ in range(48)] == expected
    assert restored._restored_pending == [prompt]


@pytest.mark.parametrize("saved,current", [(False, True), (True, False)])
def test_resume_rejects_changed_multimodal_contract(saved, current):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from rlinf.runners.async_agent_ppo_runner import AsyncAgentPPORunner

    runner = SimpleNamespace(
        cfg=OmegaConf.create(
            {"rollout": {"sglang": {"enable_shared_multi_modal_processor": current}}}
        )
    )
    with pytest.raises(ValueError, match="rollout input contract changed"):
        AsyncAgentPPORunner._load_data_state(
            runner, {"async_rollout": {"token_space": saved}}
        )
