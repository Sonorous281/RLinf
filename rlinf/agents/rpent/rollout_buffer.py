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

"""Checkpoint unconsumed RPent episodes without changing buffer admission rules."""

import asyncio
import os
from collections import deque
from pathlib import Path

import torch

from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch
from rlinf.scheduler import Channel
from rlinf.workers.rollout_buffer.rollout_buffer_worker import RolloutBufferWorker


class RPentRolloutBufferWorker(RolloutBufferWorker):
    """Keep completed groups and replay unfinished episodes in a fresh environment.

    Model checkpoints are saved before this worker's state. Already released groups
    belong to that update; fetched and issued groups remain available after resume.
    Simulator processes and partially generated episodes are never serialized.
    """

    def __init__(self, cfg, train_dataset):
        super().__init__(cfg, train_dataset)
        self._fetch_lock = asyncio.Lock()
        self._queued_requests = deque()
        self._issued_requests = {}
        self._replay_requests = deque()

    async def _run_admission(self, dataloader_channel: Channel):
        while True:
            if self._replay_requests:
                request = self._replay_requests.popleft()
            else:
                if not self._queued_requests:
                    # A checkpoint must see both the advanced dataloader and every
                    # prompt fetched from it, including prompts awaiting capacity.
                    async with self._fetch_lock:
                        batch = await self._next_batch()
                        indices = batch["idx"].tolist()
                        self.buffer.num_samples_fetched += len(indices)
                        requests = build_rollout_requests_from_batch(
                            batch,
                            group_size=self.cfg.algorithm.group_size,
                            split_size=len(batch["prompt"]),
                        )
                        self._queued_requests.extend(zip(requests, indices))
                await self.buffer.wait_for_admission_slot()
                request, sample_idx = self._queued_requests.popleft()
                group_id, _ = self.buffer.issue_group(sample_idx, self.task_name)
                request.group_ids = [group_id]
                request.task_names = [self.task_name]
                self._issued_requests[group_id] = request
            self._issued_requests = {
                gid: item
                for gid, item in self._issued_requests.items()
                if gid in self.buffer.group_tasks
            }
            await dataloader_channel.put(request, async_op=True).async_wait()

    async def save_checkpoint(self, path: str):
        """Save the dataloader, completed groups and not-yet-consumed requests."""
        async with self._fetch_lock:
            if self.buffer.pending_release_version is not None:
                raise RuntimeError(
                    "Checkpoint requires the previous training batch to be consumed"
                )
            completed = {
                gid for task in self.buffer.tasks.values() for gid in task["buffer"]
            }
            pending = {
                gid: self._issued_requests[gid]
                for gid in self.buffer.group_tasks
                if gid not in completed
            }
            state = {
                "format": "rpent-episode-buffer-v1",
                "group_size": self.cfg.algorithm.group_size,
                "task_name": self.task_name,
                "dataloader": self.buffer.dataloader.state_dict(),
                "tasks": self.buffer.tasks,
                "group_tasks": self.buffer.group_tasks,
                "next_group_id": self.buffer.next_group_id,
                "num_samples_fetched": self.buffer.num_samples_fetched,
                "queued_requests": list(self._queued_requests),
                "pending_requests": pending,
            }
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "data.pt"
            temporary = target.with_suffix(".pt.tmp")
            try:
                torch.save(state, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    async def load_checkpoint(self, path: str, start_step: int):
        """Restore completed groups once; restart unfinished episodes with the same IDs."""
        state = torch.load(
            Path(path) / "data.pt", map_location="cpu", weights_only=False
        )
        if state.get("format") != "rpent-episode-buffer-v1":
            raise ValueError(
                "Expected an RPent episode-buffer checkpoint from this branch"
            )
        if (state["group_size"], state["task_name"]) != (
            self.cfg.algorithm.group_size,
            self.task_name,
        ):
            raise ValueError(
                "Checkpoint group size and task must match the resumed run"
            )
        if len(state["group_tasks"]) > self.buffer.max_buffered_groups:
            raise ValueError(
                "Restored groups exceed rollout_buffer.max_buffered_groups"
            )
        if len(state["pending_requests"]) > self.buffer.max_inflight_groups:
            raise ValueError(
                "Restored pending episodes exceed rollout_buffer.max_inflight_groups"
            )
        loader = self.buffer.dataloader
        if self.cfg.data.shuffle:
            # Torch's sampler stores no generator state in StatefulDataLoader.
            # Recreate completed shuffles before restoring the within-epoch cursor.
            samples_per_epoch = len(loader) * loader.batch_size
            epochs = max(0, (state["num_samples_fetched"] - 1) // samples_per_epoch)
            for _ in range(epochs):
                for _ in loader.sampler:
                    pass
        loader.load_state_dict(state["dataloader"])
        self._data_iter = None
        for field in ("tasks", "group_tasks", "next_group_id", "num_samples_fetched"):
            setattr(self.buffer, field, state[field])
        self._queued_requests = deque(state["queued_requests"])
        self._issued_requests = state["pending_requests"]
        self._replay_requests = deque(self._issued_requests.values())
