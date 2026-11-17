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

"""Multimodal row alignment, cache budgets, and bounded image prefetch."""

import pytest
import torch

from rlinf.data.schema.reasoning_results import DynamicRolloutResult


@pytest.mark.parametrize("prefetch_batches", [0, 2, 8])
def test_image_prefetch_preserves_order_and_gradients(prefetch_batches, monkeypatch):
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    consumed = []
    source_ids = []
    # Simulate distinct host copies without depending on a GPU for this test.
    monkeypatch.setattr(torch.Tensor, "cpu", lambda tensor: tensor.clone())

    def batches():
        for index in range(5):
            consumed.append(index)
            ids = torch.tensor([[index + 1, 9]])
            source_ids.append(ids)
            yield {
                "input_ids": ids,
                "attention_mask": torch.ones(1, 2),
                "multi_modal_inputs": [
                    {
                        "pixel_values": torch.full((2, 3), float(index)),
                        "image_grid_thw": torch.tensor([[1, 1, 2]]),
                    }
                ],
            }

    # Each result is 48 bytes; the byte limit admits at most two batches.
    preparer = MultimodalBatchPreparer(
        "unused", prefetch_batches=prefetch_batches, prefetch_bytes=96
    )
    weight = torch.tensor(0.5, requires_grad=True)
    with preparer.iterator_context(batches()) as iterator:
        for index, batch in enumerate(iterator):
            assert batch["input_ids"] is source_ids[index]
            inputs = preparer.for_forward(batch, "cpu")
            assert torch.equal(inputs["pixel_values"], torch.full((2, 3), float(index)))
            if prefetch_batches:
                assert (
                    len(consumed) <= index + 3
                )  # Two admitted, one compact lookahead.
            (inputs["pixel_values"] * weight).sum().backward()
            assert "_prepared_multimodal_inputs" not in batch
    assert weight.grad.item() == 6 * sum(range(5))


@pytest.mark.parametrize("failure", ["prepare", "consumer", "budget"])
def test_image_prefetch_propagates_failure_and_joins_threads(failure):
    import threading

    from rlinf.utils.multimodal import MultimodalBatchPreparer

    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2),
        "multi_modal_inputs": [],
    }
    if failure == "prepare":
        batch["multi_modal_inputs"] = [
            {
                "format": "legacy",
                "pixel_values": torch.ones(1),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }
        ]
    preparer = MultimodalBatchPreparer(
        "unused",
        token_space=True,
        prefetch_batches=2,
        prefetch_bytes=1 if failure == "budget" else 4096,
    )
    error = {
        "prepare": "Legacy media",
        "budget": "exceeding",
        "consumer": "model forward",
    }[failure]
    with pytest.raises(ValueError, match=error):
        with preparer.iterator_context(iter([batch, batch])) as iterator:
            next(iterator)
            raise ValueError("model forward failed")
    assert not any(
        thread.name.startswith("rlinf-images") for thread in threading.enumerate()
    )


def test_batch_merge_pads_rows_for_a_text_only_batch():
    """A batch without images must still contribute one slot per row."""

    def batch(payloads):
        data = {
            "input_ids": torch.zeros(3, 6, dtype=torch.long),
            "idx_to_traj": [0, 1, 2],
            "prompt_lengths": torch.ones(3, dtype=torch.int32),
            "response_lengths": torch.ones(3, dtype=torch.int32),
            "is_end": torch.zeros(3, dtype=torch.bool),
        }
        if payloads is not None:
            data["multi_modal_inputs"] = payloads
        return data

    with_images = batch([None, {"tag": torch.tensor([1.0])}, None])
    for order in ([batch(None), with_images], [with_images, batch(None)]):
        merged = DynamicRolloutResult.merge_batches(
            [dict(item) for item in order], group_size=3, adjust_traj_indices=True
        )
        payloads = merged["multi_modal_inputs"]
        assert len(payloads) == merged["input_ids"].shape[0] == 6
        assert sum(entry is not None for entry in payloads) == 1


@pytest.mark.parametrize(
    "settings,token_space,error",
    [
        ({}, False, None),
        ({"cache_bytes": 1024}, False, None),
        ({"prefetch_batches": 2, "prefetch_bytes": 4096}, True, None),
        ({"cache_bytes": -1}, False, "cache_bytes"),
        ({"prefetch_batches": True}, True, "prefetch_batches"),
        ({"prefetch_bytes": 1.5}, True, "prefetch_bytes"),
        ({"prefetch_batches": 1}, True, "positive prefetch_bytes"),
        (
            {"prefetch_batches": 1, "prefetch_bytes": 4096},
            False,
            "enable_shared_multi_modal_processor",
        ),
    ],
)
def test_multimodal_learner_budgets_without_rpent(settings, token_space, error):
    from omegaconf import OmegaConf

    from rlinf.config import validate_multimodal_learner_cfg

    cfg = OmegaConf.create(
        {
            "actor": {"multimodal": settings},
            "rollout": {"sglang": {"enable_shared_multi_modal_processor": token_space}},
        }
    )
    if error:
        with pytest.raises(ValueError, match=error):
            validate_multimodal_learner_cfg(cfg)
    else:
        snapshot = OmegaConf.to_container(cfg)
        validate_multimodal_learner_cfg(cfg)
        assert OmegaConf.to_container(cfg) == snapshot


def _balance_image_rows(rank, init_method, empty_rank):
    import torch.distributed as dist

    from rlinf.utils.distributed import RolloutDataBalance

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        ids = ([] if empty_rank else [10]) if rank == 0 else [20, 21]
        batch = {
            "input_ids": torch.tensor(ids, dtype=torch.long).reshape(-1, 1),
            "prompt_lengths": torch.ones(len(ids), dtype=torch.long),
            "response_lengths": torch.ones(len(ids), dtype=torch.long),
        }
        if rank == 1:
            batch["multi_modal_inputs"] = [{"image": 20}, {"image": 21}]
        pad = {
            key: torch.zeros((1, *value.shape[1:]), dtype=value.dtype)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        partitions = [[1], [0]] if empty_rank else [[2, 0], [1, 3]]
        actual = RolloutDataBalance.from_rollout_batches_dynamic(
            batch,
            2,
            rank,
            dist.group.WORLD,
            pad,
            1,
            lambda **kwargs: partitions,
        )
        all_ids = [20, 21] if empty_rank else [10, 20, 21, 0]
        all_images = (
            [{"image": 20}, {"image": 21}]
            if empty_rank
            else [None, {"image": 20}, {"image": 21}, None]
        )
        assert actual["input_ids"].flatten().tolist() == [
            all_ids[i] for i in partitions[rank]
        ]
        assert actual["multi_modal_inputs"] == [all_images[i] for i in partitions[rank]]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("empty_rank", [False, True])
def test_dynamic_balance_keeps_images_with_rows(tmp_path, empty_rank):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_gloo_available():
        pytest.skip("Row exchange requires Gloo")
    mp.spawn(
        _balance_image_rows,
        args=((tmp_path / "balance").as_uri(), empty_rank),
        nprocs=2,
    )


@pytest.mark.parametrize("load_mtp", [None, True, False])
def test_chunked_qwen_scoring_requires_explicit_mtp_disable(load_mtp):
    from omegaconf import OmegaConf

    from rlinf.config import validate_qwen_vl_chunked_cfg

    model = {
        "model_type": "qwen3_5",
        "lm_head_chunk_size": 256,
        "pipeline_model_parallel_size": 1,
    }
    if load_mtp is not None:
        model["load_mtp"] = load_mtp
    cfg = OmegaConf.create(
        {
            "actor": {"model": model, "megatron": {"mbridge": True}},
            "algorithm": {
                "calculate_entropy": False,
                "entropy_bonus": 0,
                "sampling_params": {"temperature": 1.0},
            },
        }
    )
    if load_mtp is False:
        assert validate_qwen_vl_chunked_cfg(cfg) == 256
    else:
        with pytest.raises(ValueError, match="load_mtp=False"):
            validate_qwen_vl_chunked_cfg(cfg)
