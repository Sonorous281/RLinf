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

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture(params=[False, True], ids=["uncached", "disk-cache"])
def native_token_space(tmp_path, request):
    """Use real SGLang preprocessing when local processor assets are supplied."""
    model_path = os.environ.get("RLINF_TOKEN_SPACE_TEST_MODEL")
    if not model_path:
        pytest.skip(
            "Set RLINF_TOKEN_SPACE_TEST_MODEL to local Qwen3.5 processor assets"
        )

    from rlinf.hybrid_engines.sglang.multimodal.token_space import SGLangTokenSpace

    settings = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        "tokenizer_mode": "auto",
        "tokenizer_backend": "huggingface",
        "trust_remote_code": False,
        "revision": None,
        "model_config_parser": "auto",
        "json_model_override_args": "{}",
        "image_processor_backend": "pil",
        "disable_fast_image_processor": True,
        "mm_process_config": {},
    }
    if request.param:
        settings.update(
            enable_token_space_preprocess_cache=True,
            token_space_preprocess_cache_dir=str(tmp_path / "cache"),
            token_space_preprocess_cache_trust_url=True,
        )
    processor = SGLangTokenSpace(settings)
    try:
        yield processor
    finally:
        processor.shutdown()


@pytest.fixture
def native_multiturn(native_token_space, tmp_path):
    import io

    from PIL import Image
    from transformers import AutoConfig

    from rlinf.utils.multimodal import store_image_refs

    processor = native_token_space
    config = AutoConfig.from_pretrained(
        processor.settings["model_path"], local_files_only=True
    )
    images = []
    for mode, size, color in [
        ("RGB", (64, 96), "red"),
        ("RGBA", (96, 64), (0, 0, 255, 80)),
    ]:
        data = io.BytesIO()
        Image.new(mode, size, color).save(data, format="PNG")
        images.append(data.getvalue())
    media_directory = tmp_path / "images"
    sources = images
    if processor.cache_directory is not None:
        stored = store_image_refs(images, media_directory)
        sources = [ref["path"] for ref in stored]
    marker = [
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    ]
    first = asyncio.run(
        processor.prepare(
            [42, *marker, 43],
            sources[:1],
            {"prefix_length": 0, "image_spans": [(2, 3)]},
        )
    )
    sampled = first["prompt_token_ids"] + [44, 45]
    compact = sampled + marker + [46]
    second = asyncio.run(
        processor.prepare(
            compact,
            sources,
            {
                "prefix_length": len(sampled),
                "image_spans": [
                    *first["image_spans"],
                    (len(sampled) + 1, len(sampled) + 2),
                ],
            },
        )
    )
    assert second["prompt_token_ids"][: len(sampled)] == sampled
    assert second["image_spans"][0] == first["image_spans"][0]
    assert len(second["image_spans"]) == 2
    entry = {
        **second,
        "image_refs": store_image_refs(images, media_directory),
        "num_images": len(images),
    }
    ids = torch.tensor([second["prompt_token_ids"] + [47, 48, 0, 0]])
    mask = ids.ne(0)

    return SimpleNamespace(
        processor=processor,
        images=images,
        entry=entry,
        batch={"input_ids": ids, "attention_mask": mask, "multi_modal_inputs": [entry]},
    )


def test_native_token_space_multiturn_replay_and_cache(native_multiturn, monkeypatch):
    from sglang.srt.multimodal.cache import DiskPreprocessCache

    from rlinf.hybrid_engines.sglang.multimodal.token_space import SGLangTokenSpace
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    case = native_multiturn
    processor = case.processor
    preparer = MultimodalBatchPreparer(
        processor.settings["model_path"], token_space=True, cache_bytes=8 * 1024 * 1024
    )
    cache_hits = []
    original_get = DiskPreprocessCache.get

    def counted_get(cache, key):
        result = original_get(cache, key)
        if result is not None:
            cache_hits.append(str(cache.directory))
        return result

    monkeypatch.setattr(DiskPreprocessCache, "get", counted_get)
    first_inputs = preparer.for_forward(case.batch, "cpu")
    if processor.cache_directory is not None:
        assert cache_hits == [processor.cache_directory] * 2
        uncached = SGLangTokenSpace(
            {**processor.settings, "enable_token_space_preprocess_cache": False}
        )
        try:
            assert uncached.fingerprint == processor.fingerprint
            pixels, grid = asyncio.run(uncached.replay(case.entry, case.images))
            assert torch.equal(pixels, first_inputs["pixel_values"])
            assert torch.equal(grid, first_inputs["image_grid_thw"])
        finally:
            uncached.shutdown()
    again = preparer.for_forward(case.batch, "cpu")
    for key in first_inputs:
        assert torch.equal(first_inputs[key], again[key])
    assert first_inputs["multimodal_position_ids"].shape == (
        3,
        *case.batch["input_ids"].shape,
    )

    # Reusing cached pixels must still validate the caller's input contract.
    for key, message in [
        ("pixel_values_sha256", "pixel_values differs"),
        ("mrope_sha256", "mrope_positions differs"),
    ]:
        with pytest.raises(ValueError, match=message):
            preparer.for_forward(
                {**case.batch, "multi_modal_inputs": [{**case.entry, key: "changed"}]},
                "cpu",
            )
    Path(case.entry["image_refs"][1]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        preparer.for_forward(case.batch, "cpu")


def test_native_token_space_prefetch_matches_synchronous_replay(native_multiturn):
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    case = native_multiturn
    baseline = MultimodalBatchPreparer(
        case.processor.settings["model_path"], token_space=True
    )
    expected = baseline.for_forward(case.batch, "cpu")
    prefetch = MultimodalBatchPreparer(
        case.processor.settings["model_path"],
        token_space=True,
        prefetch_batches=2,
        prefetch_bytes=2 * sum(t.nbytes for t in expected.values()),
    )
    with prefetch.iterator_context(iter([case.batch] * 3)) as iterator:
        for batch in iterator:
            actual = prefetch.for_forward(batch, "cpu")
            assert actual.keys() == expected.keys()
            for key in expected:
                assert actual[key].device.type == "cpu"
                assert torch.equal(actual[key], expected[key])


def test_native_token_space_disk_cache_shared_across_processes(
    native_multiturn, tmp_path
):
    import subprocess

    from rlinf.hybrid_engines.sglang.multimodal.token_space import tensor_digest
    from rlinf.utils.multimodal import MultimodalBatchPreparer

    case = native_multiturn
    if case.processor.cache_directory is None:
        pytest.skip("Cross-process cache check needs the disk-cache variant")
    request = tmp_path / "replay.pt"
    result = tmp_path / "replay.json"
    torch.save(case.batch, request)
    script = """
import json, sys, torch
from pathlib import Path
from sglang.srt.multimodal.cache import DiskPreprocessCache
from rlinf.hybrid_engines.sglang.multimodal.token_space import tensor_digest
from rlinf.utils.multimodal import MultimodalBatchPreparer
hits = []
original = DiskPreprocessCache.get
def get(cache, key):
    value = original(cache, key)
    if value is not None:
        hits.append(key)
    return value
DiskPreprocessCache.get = get
batch = torch.load(sys.argv[1], weights_only=True)
model_path = batch["multi_modal_inputs"][0]["processor_settings"]["model_path"]
preparer = MultimodalBatchPreparer(model_path, token_space=True)
inputs = preparer.for_forward(batch, "cpu")
Path(sys.argv[2]).write_text(json.dumps({"hits": len(hits), "digests": {
    name: tensor_digest(value) for name, value in inputs.items()
}}))
"""
    subprocess.run(
        [sys.executable, "-c", script, str(request), str(result)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    actual = json.loads(result.read_text())
    expected = MultimodalBatchPreparer(
        case.processor.settings["model_path"], token_space=True
    ).for_forward(case.batch, "cpu")
    assert actual["hits"] == 2
    assert actual["digests"] == {
        key: tensor_digest(value) for key, value in expected.items()
    }


def test_release_capture_replays_positions_and_verifies_shared_pixels(tmp_path):
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        REPLAY_FORMAT,
        store_pixels,
    )
    from rlinf.utils.multimodal import materialize_multimodal_inputs

    pixels = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    data = {
        "format": REPLAY_FORMAT,
        "prompt_token_ids": [10, 11, 12],
        "image_spans": [[1, 2]],
        "pixel_values": pixels,
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mrope_positions": torch.tensor([[0, 1, 2]] * 3),
        "mrope_position_delta": torch.tensor([0]),
    }
    store_pixels(data, tmp_path)
    ids = torch.tensor([[10, 11, 12, 13, 0]])

    def replay():
        return materialize_multimodal_inputs(
            [data],
            "unused",
            "cpu",
            input_ids=ids,
            attention_mask=ids.ne(0),
            token_space=True,
        )

    result = replay()
    assert torch.equal(result["pixel_values"], pixels.bfloat16())
    assert result["image_input_mask"].tolist() == [[False, True, False, False, False]]
    assert result["multimodal_position_ids"][:, 0].tolist() == [[0, 1, 2, 3, 0]] * 3
    ids[0, 0] = 2
    with pytest.raises(ValueError, match="prompt"):
        replay()
    ids[0, 0] = 10
    Path(data["pixel_refs"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        replay()


def test_release_capture_checks_sampled_history_and_request_id():
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        CAPTURE_KEY,
        capture_file,
        generate_with_capture,
        save_capture,
    )

    rid = "a" * 32
    pixels = torch.ones(4, 2)

    async def generate(input_ids, image_data, sampling_params, return_logprob):
        capture = sampling_params["custom_params"][CAPTURE_KEY]
        output = SimpleNamespace(
            input_ids=input_ids,
            mrope_positions=torch.arange(len(input_ids)).repeat(3, 1),
            mrope_position_delta=0,
            mm_items=[
                SimpleNamespace(
                    offsets=[(1, 1)],
                    feature=pixels,
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                )
            ],
        )
        save_capture(capture_file(capture["ticket"], rid), output)
        return {"meta_info": {"id": rid}, "output_ids": [23]}

    def sample(ids, history=None):
        return asyncio.run(
            generate_with_capture(
                generate,
                ids,
                ["image"],
                {},
                {"image_spans": [[1, 2]], "history": history},
                False,
            )
        )

    _, _, history = sample([10, 11, 12])
    _, row, _ = sample([10, 11, 12, 23, 24], history)
    assert row["prompt_token_ids"] == [10, 11, 12, 23, 24]
    with pytest.raises(ValueError, match="sampled token history"):
        sample([10, 11, 12, 99, 24], history)
    pixels.zero_()
    with pytest.raises(ValueError, match="pixels"):
        sample([10, 11, 12, 23, 24], history)
