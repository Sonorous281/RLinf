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

"""Unit tests for the MA Megatron actor's weight-version bookkeeping.

The actor is built only in Megatron environments, so every test skips when
`megatron.core` is absent. No Ray or GPUs are launched.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.schema.reasoning_results import DynamicRolloutResult


class TestExtractVersionedTrajectoryRecords:
    """The actor's record extraction from DynamicRolloutResult objects."""

    @staticmethod
    def _make_actor():
        pytest.importorskip("megatron.core")
        # __new__ bypasses Megatron init; only the private extraction
        # method is exercised.
        from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

        return MAMegatronActor.__new__(MAMegatronActor)

    @staticmethod
    def _dynamic_result(
        num_sequence: int = 2,
        group_size: int = 1,
        version_start: list[int] | None = None,
        version_end: list[int] | None = None,
    ):
        return DynamicRolloutResult(
            num_sequence=num_sequence,
            group_size=group_size,
            idx_to_traj=list(range(num_sequence)),
            input_ids=[[1, 2, 3] for _ in range(num_sequence)],
            prompt_lengths=[2] * num_sequence,
            response_lengths=[1] * num_sequence,
            is_end=[True] * num_sequence,
            rewards=[0.0] * num_sequence,
            version_start=version_start,
            version_end=version_end,
        )

    def test_extracts_per_trajectory_records(self):
        actor = self._make_actor()
        # One result = one group (result_index 0); its trajectories are
        # trajectory 0 (two turns) and trajectory 1 (one turn).
        result = DynamicRolloutResult(
            num_sequence=3,
            group_size=2,
            idx_to_traj=[0, 0, 1],
            input_ids=[[1] * 4, [1] * 4, [1] * 4],
            prompt_lengths=[1, 1, 1],
            response_lengths=[10, 30, 20],
            is_end=[True, True, True],
            rewards=[0.0, 1.0, 2.0],
            version_start=[5, 6, 7],
            version_end=[6, 6, 7],
        )
        records = actor._extract_versioned_trajectory_records([result])
        assert len(records) == 2
        first, second = records
        # Every trajectory of one result shares the result's group index.
        assert first.group_index == 0 and second.group_index == 0
        assert first.earliest_version == 5
        assert first.completion_version == 6
        assert first.generated_tokens == 40
        assert first.call_count == 2
        assert first.cross_version_calls == 1
        assert first.reward == 0.5
        assert first.token_weighted_version_start_sum == 5 * 10 + 6 * 30
        assert second.earliest_version == 7
        assert second.generated_tokens == 20
        assert second.reward == 2.0

    def test_returns_none_without_versions(self):
        actor = self._make_actor()
        result = self._dynamic_result()
        assert actor._extract_versioned_trajectory_records([result]) is None

    def test_returns_none_if_any_result_lacks_versions(self):
        actor = self._make_actor()
        versioned = self._dynamic_result(version_start=[1, 2], version_end=[1, 2])
        assert (
            actor._extract_versioned_trajectory_records(
                [versioned, self._dynamic_result()]
            )
            is None
        )


@pytest.mark.parametrize("config", [{}, {"tools": {"rpent": None}}])
def test_text_actor_forward_does_not_require_rpent_config(config):
    pytest.importorskip("megatron.training")
    from omegaconf import OmegaConf

    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    cfg = OmegaConf.create({"rollout": {"model": {"model_path": "unused"}}, **config})
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    actor = SimpleNamespace(
        cfg=cfg,
        torch_device_type="cpu",
        multimodal_preparer=MultimodalBatchPreparer(cfg.rollout.model.model_path),
    )
    assert MAMegatronActor._forward_multimodal_inputs(actor, {}) == {}


@pytest.mark.parametrize("kind", ["actor", "critic"])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("tp", [1, 2])
def test_multimodal_right_padding_trim_preserves_outputs_and_gradients(
    monkeypatch, kind, mrope, tp
):
    """Only dense VL computation shrinks; PPO's full-width layout stays intact."""
    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )
    from rlinf.hybrid_engines.megatron.utils import preprocess_packed_seqs

    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)

    class VisionModel(torch.nn.Module):
        __module__ = "megatron.bridge.models.qwen_vl.test"
        pre_process = post_process = True

        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(20, 4)
            self.output_layer = torch.nn.Linear(4, 7 if kind == "actor" else 1)
            self.seen_shapes = []

        def forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            packed_seq_params,
            pixel_values,
            image_grid_thw,
        ):
            self.seen_shapes.append(tuple(input_ids.shape))
            assert position_ids.shape[-1] == input_ids.shape[-1]
            assert tuple(attention_mask.shape) == tuple(input_ids.shape)
            assert pixel_values.shape == (5, 4)
            assert image_grid_thw.shape == (5, 3)
            hidden = self.embedding(input_ids) + pixel_values.mean(dim=0)
            packed, _ = preprocess_packed_seqs(hidden, attention_mask)
            return self.output_layer(packed)

    torch.manual_seed(0)
    original = VisionModel()
    trimmed = copy.deepcopy(original)
    tokens = torch.arange(32).reshape(2, 16) % 20
    mask = torch.arange(16)[None] < torch.tensor([7, 11])[:, None]
    positions = torch.arange(16).expand(2, -1)
    if mrope:
        positions = positions.unsqueeze(0).expand(3, -1, -1)
    response_mask = torch.zeros_like(mask)
    response_mask[0, [3, 6]] = True
    response_mask[1, [5, 10]] = True
    label = tokens.roll(-1, dims=1)
    label_mask = response_mask.roll(-1, dims=1)
    label_mask[:, -1] = False
    pixels = torch.randn(5, 4)
    grid = torch.ones(5, 3, dtype=torch.long)

    def processor(logits, label, label_mask):
        values = (
            logits.log_softmax(-1).gather(-1, (label % 7).unsqueeze(-1)).squeeze(-1)
        )
        return {"log_probs": values.masked_fill(~label_mask, 0)}

    outputs = []
    for model, trim in [(original, False), (trimmed, True)]:
        output = MegatronModelManager.custom_forward(
            model,
            tokens,
            mask,
            positions,
            sequence_parallel=False,
            trim_right_padding=trim,
            logits_processor=processor if kind == "actor" else None,
            logits_processor_args={"label": label, "label_mask": label_mask},
            pixel_values=pixels,
            image_grid_thw=grid,
        )
        output = output["log_probs"] if kind == "actor" else output[..., 0]
        assert output.shape == (2, 16)
        # Identical token-t -> preceding-state alignment in the production workers.
        shifted = torch.nn.functional.pad(output[:, :-1], (1, 0))
        loss = (shifted * response_mask).sum()
        loss.backward()
        outputs.append(output)
    torch.testing.assert_close(outputs[0], outputs[1])
    for before, after in zip(original.parameters(), trimmed.parameters(), strict=True):
        torch.testing.assert_close(before.grad, after.grad)
    assert original.seen_shapes == [(2, 16)]
    assert trimmed.seen_shapes == [(2, 12 if tp == 2 else 11)]
    # Caller-owned targets, payload ordering and masks were never changed.
    assert tokens.shape == mask.shape == response_mask.shape == (2, 16)
