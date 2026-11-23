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
    """Accumulate per-slot score gradients and backpropagate the model once.

    Megatron completes one logical loss/backward before requesting its next slot
    (PP1, fixed MBS1). Earlier slots can therefore accumulate into detached leaves.
    At the final slot the model sees the sum of all scaled and gated gradients.
    """

    def __init__(self, scores: dict[str, torch.Tensor]):
        self.scores = scores
        self.leaves = {
            key: value.detach().requires_grad_(value.requires_grad)
            for key, value in scores.items()
        }

    def logical_scores(
        self, response_mask: torch.Tensor, *, last: bool
    ) -> dict[str, torch.Tensor]:
        """Project the original row's response mask onto next-token prediction sites."""
        if last:
            for key, leaf in self.leaves.items():
                if leaf.grad is not None:
                    self.scores[key].register_hook(
                        lambda grad, previous=leaf.grad: grad + previous
                    )
        source = self.scores if last else self.leaves
        prediction_mask = torch.nn.functional.pad(response_mask[1:], (0, 1))
        prediction_mask = prediction_mask.to(source["log_probs"].device)
        return {
            key: value[: len(response_mask)].masked_fill(~prediction_mask, 0)
            for key, value in source.items()
        }


def _score_run(first, batches, count, forward):
    rows = [first]
    for _ in range(count - 1):
        # Only the longest row needs its prefetched pixels.
        rows[-1].pop("_prepared_multimodal_inputs", None)
        rows.append(next(batches))
    longest = rows[-1]
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
    batches: Iterator[dict], forward: Callable
) -> Iterator[tuple[dict, dict]]:
    """Yield one logical score row per Megatron slot, in original collective order."""
    for batch in batches:
        count = int(batch["nested_prefix_run"].item())
        if count > 1:
            yield from _score_run(batch, batches, count, forward)
        else:
            yield batch, forward(batch)
