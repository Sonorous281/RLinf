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

"""Policy updates for RPent's synchronous GRPO cohorts and asynchronous SAO."""

import json
from pathlib import Path

from rlinf.runners.async_agent_runner import AsyncAgentRunner


class RPentRunner(AsyncAgentRunner):
    supports_actor_recompute = True

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        width = (
            self.cfg.data.sampling_batch_size if self.cfg.runner.synchronous else None
        )
        super()._build_dataloader(
            train_dataset, val_dataset, collate_fn, data_batch_size=width
        )

    def _prepare_update(self, release_handle):
        if self.cfg.runner.synchronous:
            cohort = self._wait_actor(release_handle, self._training_handles)[0]
            with self.timer("rollout/cancel_groups"):
                self.agent_loop.cancel_groups(cohort["remaining"]).wait()
            with self.timer("rollout/admit_cohort"):
                handle = self.rollout_buffer.admit_cohort(cohort)
                cohort["late_completions"] = self._wait_actor(
                    handle, self._training_handles
                )[0]
            output = (
                Path(self.cfg.runner.output_dir)
                / self.cfg.runner.experiment_name
                / "sampling"
            )
            output.mkdir(parents=True, exist_ok=True)
            (output / f"step-{self.global_steps}.json").write_text(
                json.dumps(cohort, indent=2)
            )
        elif not self.is_collocated:
            self._wait_actor(release_handle, self._training_handles)
        super()._prepare_update(release_handle)

    def _run_training(self, rollout_handle, payload_channels=None):
        if self.critic is not None or not self.recompute_logprobs:
            return super()._run_training(rollout_handle, payload_channels)
        # GRPO scores the admitted trajectories against the pre-update actor for TIS.
        with self.timer("actor/recompute_wall"):
            score = self.actor.run_inference(
                input_channel=self.rollout_channel,
                output_channel=self.actor_inference_channel,
                compute_ref_logprobs=False,
            )
            self._wait_actor(score, self._training_handles)
        with self.timer("actor/update_wall"):
            train = self.actor.run_training(
                input_channel=self.actor_inference_channel, version=self.global_steps
            )
            rollout, metrics = self._wait_actor(train, self._training_handles)[0]
        return (
            rollout,
            metrics,
            {
                "training": train.consume_duration(),
                "inference": score.consume_duration(),
            },
        )
