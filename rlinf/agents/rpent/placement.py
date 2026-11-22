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

"""RPent trains complete episode batches while rollout runs independently."""

from rlinf.utils.placement import ModelParallelComponentPlacement


class RPentComponentPlacement(ModelParallelComponentPlacement):
    """Allow sequential actor/critic updates beside dedicated rollout GPUs."""

    def _is_disaggregated(self):
        shared_learner = (
            self._critic_gpus == self._actor_gpus
            and set(self._actor_gpus).isdisjoint(self._rollout_gpus)
            and self._inference_gpus is None
            and self._critic_inference_gpus is None
        )
        return shared_learner or super()._is_disaggregated()

    @property
    def is_pipeline(self):
        # Buffer admission overlaps rollout, but each learner update consumes
        # a complete episode batch and releases its inference state afterward.
        return False
