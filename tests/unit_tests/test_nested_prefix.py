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

"""Check prefix eligibility and gradients at the original logical loss slots."""

import pytest
import torch

from rlinf.utils.image_replay import REPLAY_FORMAT
from rlinf.workers.actor.nested_prefix import NestedPrefixScores, nested_prefix_ends


def _image_row(prompt, spans):
    return {
        "format": REPLAY_FORMAT,
        "prompt_token_ids": list(range(prompt)),
        "mrope_positions": torch.arange(prompt).repeat(3, 1),
        "mrope_position_delta": torch.tensor([0]),
        "image_spans": spans,
        "image_grid_thw": torch.tensor([[1, 2, 2]] * len(spans)),
        "pixel_refs": [{"sha256": str(start)} for start, _ in spans],
        "extends_previous_turn": True,
    }


@pytest.mark.parametrize(
    "change",
    ["none", "trajectory", "token", "image", "span", "grid", "position", "rewrite"],
)
def test_nested_prefix_requires_the_same_model_inputs(change):
    rows = [torch.arange(size) for size in (5, 8, 11)]
    images = [
        _image_row(3, [(1, 2)]),
        _image_row(7, [(1, 2), (5, 6)]),
        _image_row(10, [(1, 2), (5, 6)]),
    ]
    trajectories = [3, 3, 3]
    if change == "trajectory":
        trajectories[0] = 2
    elif change == "token":
        rows[0][-1] = 42
    elif change == "image":
        images[0]["pixel_refs"][0]["sha256"] = "different"
    elif change == "span":
        images[0]["image_spans"][0] = (1, 3)
    elif change == "grid":
        images[0]["image_grid_thw"][0, 1:] = torch.tensor([1, 4])
    elif change == "position":
        images[0]["mrope_position_delta"] += 1
    elif change == "rewrite":
        images[1]["extends_previous_turn"] = False

    batch = {
        "input_ids": rows,
        "idx_to_traj": trajectories,
        "multi_modal_inputs": images,
    }
    assert nested_prefix_ends(batch) == ([2, 2, 2] if change == "none" else [0, 2, 2])


@pytest.mark.parametrize("skipped_slot", [None, 0, 2])
def test_shared_forward_preserves_masked_slot_gradients(skipped_slot):
    # Two logical rows score the same token with different weights. The last
    # slot can be gated out even when earlier slots must update the model.
    masks = [
        torch.tensor(row, dtype=torch.bool)
        for row in ([0, 0, 1, 1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1])
    ]
    token_ids = torch.tensor([0, 1, 1, 2, 0, 1, 2, 1])
    initial = torch.linspace(-0.65, 0.77, 9, dtype=torch.float64).reshape(3, 3)

    def score(weight, size):
        logits = weight[token_ids[:size]].cumsum(dim=0)
        log_probs = logits.log_softmax(-1)
        return {
            "log_probs": log_probs[:, 0],
            "entropy": -(log_probs.exp() * log_probs).sum(dim=-1),
        }

    def backward(shared):
        weight = initial.clone().requires_grad_()
        chain = NestedPrefixScores(score(weight, len(token_ids))) if shared else None
        values = []
        for slot, mask in enumerate(masks):
            prediction_indices = mask.nonzero().flatten() - 1
            if chain is None:
                scores = score(weight, len(mask))
            else:
                scores = chain.logical_scores(mask, last=slot == len(masks) - 1)
            loss = (
                scores["log_probs"][prediction_indices] * (slot - 0.5)
                - scores["entropy"][prediction_indices] * 0.03
            ).mean()
            values.append(loss.detach())
            (loss * (0 if slot == skipped_slot else 1)).backward()
        return torch.stack(values), weight.grad

    expected, actual = backward(False), backward(True)
    assert expected[1].norm() > 0
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
