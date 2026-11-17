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


"""Partial-rollout admission while a shared Actor/Critic pair updates.

Fully asynchronous actor-only training uses RolloutBufferWorker instead. This
producer retains prompt IDs and byte accounting for the SAO checkpoint contract.
"""

import asyncio
from typing import Any

from rlinf.data.schema.agent_trajectory import BufferedTrajectory, retained_size_bytes
from rlinf.scheduler import Channel


class PartialRolloutMixin:
    """Pause admission and account for completed trajectories on shared GPUs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_in_flight = 0
        self._async_sending = 0
        self._async_admission = asyncio.Event()
        self._async_admission.set()
        self._async_idle = asyncio.Event()
        self._async_idle.set()
        self._async_retained_bytes = {}
        self._async_memory_admission = asyncio.Event()
        self._async_memory_admission.set()

    async def run_async_rollout(
        self, input_channel: Channel, output_channel: Channel
    ) -> None:
        """Continuously generate complete trajectories and send them to the learner."""
        from anyio import create_task_group

        self._async_rollout_input_channel = input_channel
        capacity = len(
            range(
                self._rank,
                self.cfg.rollout.max_in_flight_rollouts,
                self._world_size,
            )
        )
        slots = asyncio.Semaphore(capacity)

        async def produce(prompt: tuple[int, list[int], Any]) -> None:
            sample_id, prompt_ids, answer = prompt
            try:
                await self._async_admission.wait()
                await self._async_memory_admission.wait()
                # A memory wait can outlast a pause requested by the learner.
                await self._async_admission.wait()
                self._async_idle.clear()
                self._async_in_flight += 1
                try:
                    result, metrics = await self.run_agentloop_rollout_group(
                        input_dict=prompt_ids,
                        answer=answer,
                        group_size=1,
                        output_channel=None,
                    )
                finally:
                    self._async_in_flight -= 1
                result.rollout_ids = [sample_id] * result.num_sequence
                self._async_sending += 1
                try:
                    entry = BufferedTrajectory(
                        sample_id, prompt_ids, answer, result, metrics
                    )
                    if self.cfg.rollout.get("max_buffer_bytes", 0):
                        self._async_retained_bytes[sample_id] = retained_size_bytes(
                            entry
                        )
                        self._update_memory_admission()
                    await output_channel.put(
                        entry,
                        async_op=True,
                    ).async_wait()
                finally:
                    self._async_sending -= 1
            finally:
                slots.release()
                if not self._async_in_flight and not self._async_sending:
                    self._async_idle.set()

        async with create_task_group() as group:
            self._async_rollout_group = group
            while True:
                await slots.acquire()
                prompt = await input_channel.get(async_op=True).async_wait()
                if prompt is None:
                    slots.release()
                    break
                group.start_soon(produce, prompt)

    async def pause_rollouts(self) -> dict[str, int]:
        """Stop new admissions and wait for active episodes and sends to finish."""
        self._async_admission.clear()
        await self._async_idle.wait()
        return self.get_async_rollout_state()

    async def resume_rollouts(self) -> None:
        """Allow queued prompts to start after the learner publishes its policy."""
        self._async_admission.set()

    def _update_memory_admission(self) -> None:
        budget = self.cfg.rollout.get("max_buffer_bytes", 0)
        if not budget:
            self._async_memory_admission.set()
            return
        # Preserve enough results to assemble one complete learner batch. A
        # byte limit must never strand the learner waiting for missing samples.
        minimum = (
            self.cfg.data.rollout_batch_size + self._world_size - 1
        ) // self._world_size
        if (
            budget
            and len(self._async_retained_bytes) >= minimum
            and sum(self._async_retained_bytes.values()) * self._world_size >= budget
        ):
            self._async_memory_admission.clear()
        else:
            self._async_memory_admission.set()

    def release_rollout_bytes(self, sample_ids: list[int]) -> None:
        """Acknowledge consumed results; retain neither their payload nor a copy."""
        for sample_id in sample_ids:
            self._async_retained_bytes.pop(sample_id, None)
        self._update_memory_admission()

    def get_async_buffer_bytes(self) -> dict:
        return {
            "retained_bytes_estimate": sum(self._async_retained_bytes.values()),
            "memory_admission_paused": int(not self._async_memory_admission.is_set()),
        }

    async def stop_rollouts(self) -> None:
        """Stop the producer and unblock its pending channel read."""
        self._async_rollout_group.cancel_scope.cancel()
        # Cancellation alone cannot unblock a Channel wait in its executor thread.
        await self._async_rollout_input_channel.put(None, async_op=True).async_wait()

    def get_async_rollout_state(self) -> dict:
        """Report producer progress for the learner's async metrics."""
        return {
            "in_flight": self._async_in_flight,
            # The learner reports this as waiting for buffer capacity.
            "waiting_for_buffer": self._async_sending,
        }
