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

from datetime import timedelta

import pytest
import torch

from rlinf.utils.image_replay import REPLAY_FORMAT
from rlinf.workers.actor.nested_prefix import (
    NestedPrefixHead,
    nested_prefix_ends,
    tp_nested_prefix_ends,
)


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


@pytest.mark.parametrize("fusion,tied", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("last", ["normal", "gated", "empty"])
def test_logical_heads_preserve_sparse_gradients(fusion, tied, last):
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    rng = torch.Generator().manual_seed(802)
    initial = torch.randn(11, 5, generator=rng)
    transform = torch.randn(5, 5, generator=rng)
    ids = torch.tensor([3, 1, 8, 7, 0, 4, 6, 2])
    labels = torch.cat((ids[1:], ids.new_zeros(1)))
    masks = [
        torch.tensor(row, dtype=torch.bool)
        for row in ([0, 0, 1, 1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1])
    ]
    if last == "empty":
        masks[-1].zero_()

    def run(shared):
        embedding = initial.clone().requires_grad_()
        weight = embedding if tied else initial.flip(0).requires_grad_()
        bias = torch.zeros(len(initial), requires_grad=True)
        body = transform.clone().requires_grad_()
        if fusion:
            weight.main_grad = torch.zeros_like(weight)
            weight.grad_added_to_main_grad = False
        body_backwards = []
        calls = []

        def decoder(length):
            hidden = (embedding[ids[:length]].cumsum(0) @ body).unsqueeze(0)
            hidden.register_hook(lambda grad: body_backwards.append(length))
            return hidden

        def project(hidden, *, labels, label_mask):
            calls.append(labels.numel())
            return chunked_vocab_parallel_log_probs(
                hidden,
                weight,
                labels,
                label_mask,
                chunk_size=2,
                temperature=0.9,
                bias=bias,
                calculate_entropy=True,
                gradient_accumulation_fusion=fusion,
            )

        chain = (
            NestedPrefixHead(decoder(len(ids)), labels[None], project)
            if shared
            else None
        )
        results = []
        for slot, mask in enumerate(masks):
            next_mask = torch.nn.functional.pad(mask[1:], (0, 1))
            if chain is None:
                scores = project(
                    decoder(len(mask))[0],
                    labels=labels[: len(mask)],
                    label_mask=next_mask,
                )
            else:
                scores = chain.logical_scores(mask, last=slot == len(masks) - 1)
            results.append({k: v[next_mask].detach() for k, v in scores.items()})
            # The final backward must deliver prior gradients even with zero loss or zero targets.
            loss = scores["log_probs"].sum() * (
                0 if last == "gated" and slot == 2 else slot - 0.7
            )
            loss.backward()
            if shared and slot < 2:
                assert not body_backwards and body.grad is None
        if shared:
            assert body_backwards == [len(ids)]
            assert calls == [mask.sum().item() for mask in masks]
            assert not chain.pending
        weight_grad = weight.grad + weight.main_grad if fusion else weight.grad
        return (
            results,
            body.grad,
            weight_grad,
            bias.grad,
            None if tied else embedding.grad,
        )

    torch.testing.assert_close(run(True), run(False), atol=2e-5, rtol=2e-5)


def _check_tp_schedules(rank, rendezvous):
    dist = torch.distributed
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        rows = [torch.arange(size) for size in (5, 8)]
        images = [_image_row(3, []), _image_row(7, [])]
        batch = {"input_ids": rows, "idx_to_traj": [0, 0], "multi_modal_inputs": images}
        tp_nested_prefix_ends(batch, dist.group.WORLD, "cpu")
        images[1]["extends_previous_turn"] = rank == 0
        with pytest.raises(RuntimeError, match="Tensor-parallel ranks disagree"):
            tp_nested_prefix_ends(batch, dist.group.WORLD, "cpu")
        images[1]["extends_previous_turn"] = False
        if rank:
            images[1]["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        with pytest.raises(RuntimeError, match="Tensor-parallel ranks disagree"):
            tp_nested_prefix_ends(batch, dist.group.WORLD, "cpu")
        images[1]["extends_previous_turn"] = True
        if rank:
            batch["response_mask"] = [None]
        with pytest.raises((ValueError, RuntimeError), match="nested prefix"):
            tp_nested_prefix_ends(batch, dist.group.WORLD, "cpu")
    finally:
        dist.destroy_process_group()


def test_tp_peers_reject_different_prefix_schedules(tmp_path):
    torch.multiprocessing.spawn(
        _check_tp_schedules, args=((tmp_path / "tp").as_uri(),), nprocs=2
    )


def test_nested_head_requires_backward_before_the_next_slot():
    hidden = torch.ones(1, 5, 2, requires_grad=True)
    labels = torch.zeros(1, 5, dtype=torch.long)
    head = NestedPrefixHead(hidden, labels, lambda h, **kw: {"log_probs": h.sum(-1)})
    mask = torch.tensor([False, True, True])
    head.logical_scores(mask, last=False)
    with pytest.raises(RuntimeError, match="did not finish backward"):
        head.logical_scores(mask, last=True)
