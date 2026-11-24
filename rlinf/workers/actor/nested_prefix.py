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

"""Score a causal prefix once while retaining each logical loss microbatch."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlinf.data.schema.actor_batch import DynamicActorBatch


def _extends_prefix(short_ids, long_ids, short, long) -> bool:
    """Require identical token, image and rotary-position inputs at every shared token."""
    from rlinf.utils.image_replay import REPLAY_FORMAT, replay_positions

    if not (
        short
        and long
        and short.get("format") == long.get("format") == REPLAY_FORMAT
        and long.get("extends_previous_turn")
        and "pixel_refs" in short
        and "pixel_refs" in long
        and len(short_ids) < len(long_ids)
        and len(short_ids) <= len(long["prompt_token_ids"])
        and torch.equal(short_ids, long_ids[: len(short_ids)])
    ):
        return False
    spans = [span for span in long["image_spans"] if span[0] < len(short_ids)]
    count = len(spans)
    if (
        spans != short["image_spans"]
        or not torch.equal(short["image_grid_thw"], long["image_grid_thw"][:count])
        or len(short["pixel_refs"]) != count
        or len(long["pixel_refs"]) != len(long["image_spans"])
        or [ref["sha256"] for ref in short["pixel_refs"]]
        != [ref["sha256"] for ref in long["pixel_refs"][:count]]
    ):
        return False
    return torch.equal(
        replay_positions(short, short_ids.tolist()),
        long["mrope_positions"][:, : len(short_ids)],
    )


def nested_prefix_ends(batch: DynamicActorBatch) -> list[int]:
    """Return the inclusive end of each consecutive, input-equivalent prefix chain."""
    rows = batch["input_ids"]
    ends = list(range(len(rows)))
    images = batch.get("multi_modal_inputs")
    if images is None:
        return ends
    trajectories = batch["idx_to_traj"]
    for i in reversed(range(len(rows) - 1)):
        if trajectories[i] == trajectories[i + 1] and _extends_prefix(
            rows[i], rows[i + 1], images[i], images[i + 1]
        ):
            ends[i] = ends[i + 1]
    return ends


class NestedPrefixScores:
    """Project the scores from a shared inference pass onto each logical row."""

    def __init__(self, scores: dict[str, torch.Tensor]):
        self.scores = scores

    def logical_scores(
        self, response_mask: torch.Tensor, *, last: bool
    ) -> dict[str, torch.Tensor]:
        prediction_mask = torch.nn.functional.pad(response_mask[1:], (0, 1))
        prediction_mask = prediction_mask.to(self.scores["log_probs"].device)
        return {
            key: value[: len(response_mask)].masked_fill(~prediction_mask, 0)
            for key, value in self.scores.items()
        }


class _SelectPrefixHidden(torch.autograd.Function):
    """Keep sparse head gradients until the final slot can backpropagate the body."""

    @staticmethod
    def forward(ctx, hidden, indices, pending, last):
        ctx.save_for_backward(indices)
        ctx.hidden_shape = hidden.shape
        ctx.pending = pending
        ctx.last = last
        return hidden.index_select(0, indices)

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        if not ctx.last:
            ctx.pending.append((indices, grad.detach()))
            return None, None, None, None

        total = grad.new_zeros(ctx.hidden_shape)
        for earlier_indices, earlier_grad in ctx.pending:
            total.index_add_(0, earlier_indices, earlier_grad)
        ctx.pending.clear()
        total.index_add_(0, indices, grad)
        return total, None, None, None


class NestedPrefixHead:
    """Project each logical slot using its original weights and scored positions."""

    def __init__(self, hidden, labels, score):
        # Fixed MBS1. SP/TP synchronization wraps the physical hidden tensor once.
        self.hidden = hidden[0]
        self.labels = labels[0]
        self.score = score
        self.pending = []

    def logical_scores(self, response_mask, *, last):
        indices = response_mask[1:].nonzero(as_tuple=True)[0].to(self.hidden.device)
        hidden = self.hidden
        if not last:
            # Earlier backward calls must not enter or free the physical model graph.
            hidden = hidden.detach().requires_grad_(hidden.requires_grad)
        selected = _SelectPrefixHidden.apply(hidden, indices, self.pending, last)
        targets = self.labels.index_select(0, indices)
        scores = self.score(selected, labels=targets, label_mask=None)
        return {
            key: values.new_zeros(len(response_mask)).index_copy(0, indices, values)
            for key, values in scores.items()
        }


def _score_run(first, batches, count, forward, forward_head):
    rows = [first]
    rows.extend(next(batches) for _ in range(count - 1))
    longest = rows[-1]
    if forward_head is not None:
        shared = forward_head(longest)
    else:
        mask = longest["response_mask"][0].clone()
        for row in rows[:-1]:
            other = row["response_mask"][0]
            mask[: len(other)] |= other
        physical = forward({**longest, "response_mask": [mask]})
        shared = NestedPrefixScores({key: value[0] for key, value in physical.items()})
    for i, row in enumerate(rows):
        logical = shared.logical_scores(row["response_mask"][0], last=i == count - 1)
        yield row, {key: [value] for key, value in logical.items()}


def score_nested_microbatches(
    batches: Iterator[dict], forward: Callable, forward_head: Callable | None = None
) -> Iterator[tuple[dict, dict]]:
    """Yield one logical score row per Megatron slot, in original collective order."""
    for batch in batches:
        count = int(batch["nested_prefix_run"].item())
        if count > 1:
            yield from _score_run(batch, batches, count, forward, forward_head)
        else:
            yield batch, forward(batch)
