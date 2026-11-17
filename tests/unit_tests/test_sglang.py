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

"""SGLang request metadata, image processing, and service lifecycle."""

import asyncio
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def sglang_args_modules(monkeypatch):
    """Exercise argument assembly without importing the vendor GPU engine."""
    import dataclasses

    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.5.19" if name == "sglang" else original_version(name),
    )
    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    # Vendor capability checks run against CPU-only stand-ins for its APIs.
    for module_name, class_name, names in (
        ("server_args", "ServerArgs", ["enable_token_space_processor"]),
        (
            "managers.io_struct",
            "GenerateReqInput",
            ["mm_token_expansion_start_len", "return_prompt_token_ids"],
        ),
    ):
        monkeypatch.setitem(
            sys.modules,
            f"sglang.srt.{module_name}",
            SimpleNamespace(
                **{class_name: dataclasses.make_dataclass(class_name, names)}
            ),
        )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.multimodal.token_space_mm_process.processor",
        SimpleNamespace(
            TokenSpaceProcessor=SimpleNamespace(
                process=lambda mm_token_expansion_start_len: None
            )
        ),
    )
    # The backend owns vendor engine/process creation; no backend is launched here.
    monkeypatch.setitem(
        sys.modules,
        "rlinf.workers.rollout.sglang.backends",
        SimpleNamespace(make_backend=None),
    )
    directory = Path(__file__).parents[2] / "rlinf/workers/rollout/sglang"
    modules = []
    for filename in ("__init__", "sglang_worker", "process"):
        name = "rlinf.workers.rollout.sglang"
        if filename != "__init__":
            name += f".{filename}"
        spec = importlib.util.spec_from_file_location(
            name, directory / f"{filename}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@pytest.fixture
def sglang_args_worker():
    from omegaconf import OmegaConf

    rollout = OmegaConf.create(
        {
            "enforce_eager": False,
            "validate_weight": False,
            "validate_weight_first_sync": True,
            "model": {
                "model_path": "/model",
                "precision": "bf16",
                "trust_remote_code": True,
            },
            "sglang": {
                "use_torch_compile": False,
                "torch_compile_max_bs": 4,
                "decode_log_interval": 100,
                "attention_backend": "triton",
                "disable_overlap_schedule": True,
                "chunked_prefill_size": 4096,
            },
            "cuda_graph_max_bs": 4,
            "max_running_requests": 2,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
            "detokenize": False,
        }
    )
    return SimpleNamespace(
        _cfg=SimpleNamespace(runner=SimpleNamespace(resume_dir=None)),
        _cfg_rollout=rollout,
        _rank=0,
        weight_reload="sync",
        acquire_free_port=lambda: 12345,
        log_on_first_rank=lambda message: None,
    )


def test_sglang_server_args_with_installed_vendor(sglang_args_worker):
    pytest.importorskip("sglang.srt.server_args")
    model = os.environ.get("RLINF_TEST_SGLANG_MODEL")
    if not model:
        pytest.skip("Set RLINF_TEST_SGLANG_MODEL to a local model directory")
    if not torch.cuda.is_available():
        pytest.skip("The native SGLang Engine import requires a CUDA device")
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    worker = sglang_args_worker
    worker._cfg_rollout.model.model_path = model
    worker._cfg_rollout.enforce_eager = True
    args = SGLangWorker._build_server_args(worker)
    graph_limit = args.get("cuda_graph_max_bs", args.get("cuda_graph_max_bs_decode"))
    assert graph_limit == 2
    assert args["chunked_prefill_size"] == 4096
    assert args["load_format"] == "auto"
    assert not args.get("enable_token_space_processor", False)


@pytest.mark.parametrize(
    "args_api,backup,resume_dir,processor_key,enabled",
    [
        ("dataclass-old", False, None, None, False),
        ("dataclass-new", True, "/checkpoint", None, False),
        ("struct", False, "/checkpoint", None, False),
        ("dataclass-new", False, None, "enable_shared_multi_modal_processor", True),
        ("dataclass-new", False, None, "enable_token_space_processor", True),
        ("struct", False, None, "enable_shared_multi_modal_processor", False),
    ],
)
def test_sglang_server_args_preserve_loading_and_processor_settings(
    monkeypatch,
    resume_dir,
    backup,
    args_api,
    processor_key,
    enabled,
    sglang_args_modules,
    sglang_args_worker,
):
    import dataclasses

    _, sglang_worker, _ = sglang_args_modules

    # Isolate the vendor engine boundary; exercise RLinf's real argument assembly.
    def server_args(**kwargs):
        if args_api == "struct":
            import msgspec

            def resolve_once(self):
                self.chunked_prefill_size //= 2

            cls = msgspec.defstruct(
                "ServerArgs",
                [(key, object) for key in kwargs],
                namespace={
                    "resolve_once": resolve_once,
                    "resolved_dict": lambda self: msgspec.structs.asdict(self),
                },
            )
            return cls(**kwargs)
        cls = dataclasses.make_dataclass(
            "ServerArgs", [(key, object) for key in kwargs]
        )
        return cls(**kwargs)

    monkeypatch.setattr(sglang_worker, "ServerArgs", server_args)
    graph_field = (
        "cuda_graph_max_bs"
        if args_api == "dataclass-old"
        else "cuda_graph_max_bs_decode"
    )
    monkeypatch.setattr(
        sglang_worker, "_SERVER_ARGS_FIELDS", {graph_field, "chunked_prefill_size"}
    )
    worker = sglang_args_worker
    worker._cfg.runner.resume_dir = resume_dir
    worker._cfg_rollout.sglang.enable_weights_cpu_backup = backup
    if processor_key is not None:
        worker._cfg_rollout.sglang[processor_key] = enabled
    args = sglang_worker.SGLangWorker._build_server_args(worker)
    assert args["disable_overlap_schedule"] is True
    assert args["enable_weights_cpu_backup"] is bool(backup)
    assert args["load_format"] == ("auto" if backup or resume_dir is None else "dummy")
    assert args[graph_field] == 2
    assert args["chunked_prefill_size"] == 4096
    assert "enable_shared_multi_modal_processor" not in args
    assert args.get("enable_token_space_processor", False) is enabled
    if enabled:
        assert args["disable_fast_image_processor"] is True
        assert args["image_processor_backend"] == "pil"
    else:
        assert "enable_token_space_processor" not in args
        assert "disable_fast_image_processor" not in args
        assert "image_processor_backend" not in args


@pytest.mark.parametrize(
    "settings,enabled,error",
    [
        ({}, False, None),
        ({"enable_shared_multi_modal_processor": True}, True, None),
        ({"enable_token_space_processor": True}, True, None),
        ({"enable_shared_multi_modal_processor": False}, False, None),
        ({"enable_shared_multi_modal_processor": None}, None, "must be a boolean"),
        ({"enable_token_space_processor": "false"}, None, "must be a boolean"),
        (
            {
                "enable_shared_multi_modal_processor": True,
                "enable_token_space_processor": True,
            },
            None,
            "keep only",
        ),
        (
            {
                "enable_shared_multi_modal_processor": False,
                "enable_token_space_processor": True,
            },
            None,
            "keep only",
        ),
    ],
)
def test_shared_multimodal_switch_accepts_legacy_configs(settings, enabled, error):
    from omegaconf import OmegaConf

    from rlinf.config import (
        shared_multi_modal_processor_enabled,
        validate_sglang_multimodal_cfg,
    )

    cfg = OmegaConf.create(settings)
    OmegaConf.set_struct(cfg, True)
    if error:
        with pytest.raises(ValueError, match=error):
            validate_sglang_multimodal_cfg(cfg)
    else:
        validate_sglang_multimodal_cfg(cfg)
        assert shared_multi_modal_processor_enabled(cfg) is enabled
    assert OmegaConf.to_container(cfg) == settings


@pytest.mark.parametrize("release", ["0.5.12.post1", "0.5.19", "0.5.20.dev0"])
def test_sglang_experimental_build_requires_explicit_rollout(
    release, sglang_args_modules
):
    from packaging.version import parse

    package, _, process = sglang_args_modules
    package.sglang_version = parse(release)
    package.require_supported_sglang(token_space=True)
    if release != "0.5.20.dev0":
        package.require_supported_sglang()
        return
    with pytest.raises(ValueError, match="enable_shared_multi_modal_processor=true"):
        package.require_supported_sglang()
    for server_type in ("srt", "embodied"):
        server = process.SGLangServerProcess(
            {"enable_token_space_processor": True},
            acquire_free_port=lambda **kwargs: pytest.fail("allocated a port"),
            server_type=server_type,
        )
        with pytest.raises(ValueError, match="supported release"):
            server.start()
        assert server._server_proc is None


def test_token_space_processor_resolution_preserves_launch_inputs(monkeypatch):
    from rlinf.hybrid_engines.sglang.multimodal.token_space import processor_settings

    class ServerArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def resolve_once(self):
            self.kwargs["tokenizer_path"] = self.kwargs["model_path"]
            self.kwargs["chunked_prefill_size"] //= 2
            self.kwargs["mm_process_config"]["image"] = {"max_pixels": 4096}

        def resolved_dict(self):
            return self.kwargs

    monkeypatch.setitem(
        sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=ServerArgs)
    )
    raw = {
        "model_path": "/model",
        "tokenizer_path": None,
        "tokenizer_mode": "auto",
        "tokenizer_backend": "huggingface",
        "trust_remote_code": False,
        "revision": None,
        "model_config_parser": "auto",
        "json_model_override_args": "{}",
        "image_processor_backend": "pil",
        "disable_fast_image_processor": True,
        "mm_process_config": {},
        "chunked_prefill_size": 4096,
    }
    settings = processor_settings(raw)
    assert settings["tokenizer_path"] == "/model"
    assert settings["mm_process_config"] == {"image": {"max_pixels": 4096}}
    assert raw["tokenizer_path"] is None
    assert raw["chunked_prefill_size"] == 4096
    assert raw["mm_process_config"] == {}


def test_sglang_packet_metadata_stays_aligned_with_cumulative_tokens():
    from rlinf.hybrid_engines.sglang.common.token_metadata import (
        accumulate_token_metadata,
        decode_generation_result,
    )

    states = {"episode": SimpleNamespace()}
    for version, count, forced in [(1, 3, False), (2, 2, False), (2, 1, True)]:
        packet = SimpleNamespace(
            rids=["episode", "cancelled"],
            customized_info={
                "rollout_weight_versions": [[version] * count, [0]],
                "forced_token_mask": [[forced] * count, [False]],
            },
        )
        accumulate_token_metadata(packet, states)
    assert packet.customized_info["rollout_weight_versions"][0] == [1, 1, 1, 2, 2, 2]
    assert packet.customized_info["forced_token_mask"][0] == [False] * 5 + [True]

    result = decode_generation_result(
        {
            "output_ids": [10, 11, 12, 13, 14, 15],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 8,
                "forced_token_mask": packet.customized_info["forced_token_mask"][0],
                "rollout_weight_versions": packet.customized_info[
                    "rollout_weight_versions"
                ][0],
                "output_token_logprobs": [
                    (-0.5, token, None) for token in range(10, 16)
                ],
            },
        },
        return_logprobs=True,
    )
    assert result["forced_token_positions"] == [5]
    assert (
        len(result["logprobs"])
        == len(result["weight_versions"])
        == len(result["output_ids"])
    )


def test_token_space_http_and_generation_metadata_preserve_prompt_ids(monkeypatch):
    from rlinf.hybrid_engines.sglang.common.token_metadata import (
        decode_generation_result,
    )
    from rlinf.utils.http_client import InferenceHTTPClient

    client = InferenceHTTPClient("http://127.0.0.1:1")

    async def capture_request(path, body):
        assert path == "/generate"
        return body

    monkeypatch.setattr(client, "_apost", capture_request)
    body = asyncio.run(
        client.async_generate(
            input_ids=[1, 9, 9, 2],
            sampling_params={"max_new_tokens": 2},
            return_logprob=True,
            image_data=[b"image"],
            return_prompt_token_ids=True,
            mm_token_expansion_start_len=4,
        )
    )
    assert body["return_prompt_token_ids"] is True
    assert body["mm_token_expansion_start_len"] == 4
    assert body["image_data"] == ["aW1hZ2U="]
    result = decode_generation_result(
        {
            "output_ids": [3],
            "prompt_token_ids": body["input_ids"],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 4,
                "output_token_logprobs": [(-0.5, 3, None)],
            },
        },
        return_logprobs=True,
    )
    assert result["prompt_token_ids"] == [1, 9, 9, 2]
    assert result["logprobs"] == [-0.5]
    legacy = InferenceHTTPClient._generate_body(None, [1, 2], {}, False)
    assert "return_prompt_token_ids" not in legacy
    assert "mm_token_expansion_start_len" not in legacy


@pytest.mark.parametrize("namespace_package", [False, True])
def test_bundled_cudart_preserves_library_search_paths(
    tmp_path, monkeypatch, namespace_package, sglang_args_modules
):
    _pin_bundled_cudart = sglang_args_modules[2]._pin_bundled_cudart

    # Isolate the installed wheel layout, including namespace-only CUDA wheels.
    nvidia = ModuleType("nvidia")
    runtime = ModuleType("nvidia.cuda_runtime")
    runtime.__path__ = [str(tmp_path)]
    runtime.__file__ = None if namespace_package else str(tmp_path / "__init__.py")
    nvidia.cuda_runtime = runtime
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)
    monkeypatch.setitem(sys.modules, "nvidia.cuda_runtime", runtime)
    library = tmp_path / "lib"
    library.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/existing:{library}:/another")
    _pin_bundled_cudart()
    assert os.environ["LD_LIBRARY_PATH"] == f"{library}:/existing:/another"


@pytest.fixture
def qwen_image_processor(monkeypatch):
    """Real SGLang/HF preprocessing; model assets are optional in CPU CI."""
    model_path = os.environ.get("RLINF_TEST_QWEN_VL_MODEL")
    if not model_path:
        pytest.skip("Set RLINF_TEST_QWEN_VL_MODEL to a local Qwen3.5 processor.")
    pytest.importorskip("sglang")
    from transformers import AutoConfig, AutoProcessor

    from rlinf.hybrid_engines.sglang.multimodal.qwen_vl import TokenizedQwenVLProcessor

    hf = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    server = SimpleNamespace(
        mm_process_config={},
        skip_tokenizer_init=True,
        disable_fast_image_processor=True,
        keep_mm_feature_on_device=False,
        tokenizer_worker_num=1,
    )
    # Recent SGLang reads processor settings from a process-local context.
    # Publish real config bags without initializing the GPU engine.
    if importlib.util.find_spec("sglang.srt.runtime_context") is not None:
        from sglang.srt import runtime_context
        from sglang.srt.server_args import ServerArgs

        server = ServerArgs(
            model_path=model_path,
            mm_process_config={},
            skip_tokenizer_init=True,
            disable_fast_image_processor=True,
            mm_preprocess_cache_size_mb=0,
        )
        context = runtime_context.RuntimeContext(runtime_context.ParallelContext())
        context.set_server_args(server)
        monkeypatch.setattr(runtime_context, "_CONTEXT", context)
    processor = TokenizedQwenVLProcessor(config, server, hf, "cpu", skip_mm_pool=True)
    try:
        yield processor, hf, config
    finally:
        processor.io_executor.shutdown()
        processor.cpu_executor.shutdown()


@pytest.mark.parametrize(
    ("detokenize", "parts"),
    [(False, ["\n", "\n"]), (True, ["\n", "\n"]), (False, ["a", "hello"])],
)
def test_qwen_image_processor_preserves_sampled_tokens(
    qwen_image_processor, detokenize, parts
):
    from PIL import Image
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    processor, hf, config = qwen_image_processor
    processor.server_args.skip_tokenizer_init = not detokenize
    processor.skip_tokenizer_init = not detokenize
    images = [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (96, 64), "blue")]
    separator = "".join(parts)
    text = f"A<|vision_start|><|image_pad|><|vision_end|>{separator}<|im_end|>B<|vision_start|><|image_pad|><|vision_end|>C"
    expected = hf(text=[text], images=images, return_tensors="pt")
    canonical = expected.input_ids[0].tolist()
    at = canonical.index(config.vision_end_token_id) + 1
    separator_size = len(hf.tokenizer.encode(separator, add_special_tokens=False))
    prompt = (
        canonical[:at]
        + [
            token
            for part in parts
            for token in hf.tokenizer.encode(part, add_special_tokens=False)
        ]
        + canonical[at + separator_size :]
    )
    assert hf.tokenizer.decode(prompt) == hf.tokenizer.decode(canonical)
    assert prompt != hf.tokenizer.encode(
        hf.tokenizer.decode(prompt), add_special_tokens=False
    )
    request = SimpleNamespace(
        video_data=None, audio_data=None, rid="token-preservation"
    )
    output = asyncio.run(processor.process_mm_data_async(images, prompt, request))
    assert output.input_ids == prompt
    assert torch.equal(
        torch.cat([item.feature for item in output.mm_items]), expected.pixel_values
    )
    input_ids = torch.tensor([prompt])
    for item in output.mm_items:
        for start, end in item.offsets:
            assert input_ids[0, start : end + 1].eq(config.image_token_id).all()
    # HF's independent position implementation needs only configuration, not weights.
    geometry = SimpleNamespace(config=config)
    geometry.get_vision_position_ids = MethodType(
        Qwen3_5Model.get_vision_position_ids, geometry
    )
    positions, delta = Qwen3_5Model.get_rope_index(
        geometry,
        input_ids,
        (input_ids == config.image_token_id).int(),
        image_grid_thw=expected.image_grid_thw,
    )
    assert torch.equal(output.mrope_positions, positions[:, 0])
    assert torch.equal(output.mrope_position_delta, delta)


@pytest.mark.parametrize("invalid", ["grid", "image_count"])
def test_qwen_image_processor_rejects_inconsistent_images(
    qwen_image_processor, invalid
):
    from PIL import Image

    processor, hf, config = qwen_image_processor
    images = [Image.new("RGB", (64, 64), "red")]
    prompt = (
        hf(
            text=["A<|vision_start|><|image_pad|><|vision_end|>B"],
            images=images,
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )
    if invalid == "grid":
        prompt.remove(config.image_token_id)
        message = "processor grid needs"
    else:
        images.append(images[0])
        message = "image spans, but received"
    request = SimpleNamespace(video_data=None, audio_data=None, rid="invalid-image")
    with pytest.raises(ValueError, match=message):
        asyncio.run(processor.process_mm_data_async(images, prompt, request))


def test_qwen_image_processor_keeps_text_request_behavior(qwen_image_processor):
    from PIL import Image

    processor, hf, _ = qwen_image_processor
    images = [Image.new("RGB", (64, 64), "red")]
    text = "A<|vision_start|><|image_pad|><|vision_end|>\n\nB"
    expected = hf(text=[text], images=images, return_tensors="pt")
    request = SimpleNamespace(video_data=None, audio_data=None, rid="text-request")
    output = asyncio.run(processor.process_mm_data_async(images, text, request))
    assert output.input_ids == expected.input_ids[0].tolist()
    assert torch.equal(output.mm_items[0].feature, expected.pixel_values)


@pytest.mark.parametrize("failure", [None, "live_kv", "flush"])
def test_sglang_pause_preserves_tokens_and_fails_closed(monkeypatch, failure):
    from array import array

    vendor = pytest.importorskip("sglang.srt.managers.scheduler")
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams

    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler

    active = Req(
        "active",
        "",
        array("q", [101, 102]),
        SamplingParams(max_new_tokens=5),
        return_logprob=True,
    )
    queued = Req(
        "queued",
        "",
        array("q", [101, 102]),
        SamplingParams(max_new_tokens=5),
        return_logprob=True,
    )
    kv = getattr(active, "kv", active)
    active.output_ids.extend([103, 104])
    getattr(active, "logprob", active).output_token_logprobs_val = [-1.2, -0.9]
    active.customized_info = {"rollout_weight_versions": [5, 5]}
    kv.req_pool_idx = 4
    kv.kv_allocated_len = 4
    active.logprob_start_len = queued.logprob_start_len = 2

    scheduler = object.__new__(Scheduler)
    scheduler.__dict__.update(
        server_args=SimpleNamespace(),
        _rlinf_parked=None,
        _rlinf_update_incomplete=False,
        _rlinf_weight_version=5,
        _rlinf_generated_tokens=2,
        _rlinf_retractions=0,
        _rlinf_worker=SimpleNamespace(
            torch_platform=SimpleNamespace(synchronize=lambda: None)
        ),
        running_batch=ScheduleBatch(reqs=[active]),
        waiting_queue=[queued],
        last_batch=None,
        chunked_req=None,
        disaggregation_mode=vendor.DisaggregationMode.NULL,
        enable_overlap=False,
        enable_overlap_mlx=False,
        decode_offload_manager=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        _pause_started_at=None,
        _pause_intervals=[],
        metrics_reporter=SimpleNamespace(
            last_gen_throughput=1.0, current_scheduler_metrics_enabled=False
        ),
        kv_events_publisher=SimpleNamespace(publish_kv_events=lambda: None),
        enable_priority_scheduling=False,
        enable_hicache_storage=False,
        enable_lmcache=False,
        max_queued_requests=None,
        processed_tokens_counter=7,
    )

    def release_gpu_rows(*, reqs, **kwargs):
        assert reqs == [active] and kv.req_pool_idx == 4
        if failure == "live_kv":
            if not hasattr(active, "kv"):
                raise RuntimeError("GPU release failed")
            return
        kv.req_pool_idx = None
        kv.kv_allocated_len = 0
        active.reset_for_retract()

    def flush_gpu_cache():
        assert scheduler._engine_paused
        assert scheduler.running_batch.is_empty() and not scheduler.waiting_queue
        assert kv.req_pool_idx is None and kv.kv_allocated_len == 0
        return failure != "flush"

    # Only the vendor GPU release and flush are substituted; queue operations
    # run through the installed vendor scheduler and RLinf adapter.
    if hasattr(active, "kv"):
        monkeypatch.setattr(vendor, "retract_all", release_gpu_rows)
    else:

        def release_req(batch, index, remaining, server_args):
            release_gpu_rows(reqs=[batch.reqs[index]])

        monkeypatch.setattr(ScheduleBatch, "release_req", release_req)
    monkeypatch.setattr(vendor.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(scheduler, "flush_cache", flush_gpu_cache)
    if failure:
        with pytest.raises(RuntimeError):
            scheduler.pause_for_weight_update()
        assert scheduler._engine_paused
        assert scheduler._rlinf_weight_version == 5
        with pytest.raises(RuntimeError):
            scheduler.pause_for_weight_update()
        with pytest.raises(RuntimeError):
            scheduler.resume_after_weight_update()
        return

    paused = scheduler.pause_for_weight_update()
    assert paused["paused"] and paused["parked_requests"] == 2
    assert scheduler.pause_for_weight_update() == paused
    state = scheduler.get_async_rollout_state(include_requests=True)
    assert state["requests"]["active"] == {
        "output_ids": [103, 104],
        "logprobs": [-1.2, -0.9],
        "weight_versions": [5, 5],
    }
    assert active.logprob_start_len == queued.logprob_start_len == -1
    scheduler._rlinf_weight_version = 6
    assert not scheduler.resume_after_weight_update()["paused"]
    assert [req.rid for req in scheduler.waiting_queue] == ["active", "queued"]
    active.output_ids.append(105)
    scheduler._stamp_generated_weight_versions(active)
    assert active.customized_info["rollout_weight_versions"] == [5, 5, 6]


def test_sglang_failed_weight_load_requires_engine_restart(monkeypatch):
    pytest.importorskip("sglang.srt.managers.scheduler")
    from rlinf.hybrid_engines.sglang.common.io_struct import SyncHFWeightInput
    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler
    from rlinf.scheduler import Worker
    from rlinf.utils.placement import RolloutSyncMode

    class FailingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def load_weights(self, weights):
            self.weight.data.copy_(weights[0][1])
            raise RuntimeError("interrupted weight transfer")

    model = FailingModel()
    scheduler = object.__new__(Scheduler)
    scheduler.__dict__.update(
        _rlinf_update_incomplete=False,
        _rlinf_weight_version=5,
        _rlinf_parked={},
        weight_reload="sync",
        rollout_sync_mode=RolloutSyncMode.DISAGGREGATED,
        is_weight_offloaded=False,
        _actor_group_name="actor",
        actor_weight_rank=0,
        _rlinf_worker=SimpleNamespace(recv=lambda **kwargs: {"weight": torch.ones(1)}),
        tp_worker=SimpleNamespace(
            worker=SimpleNamespace(model_runner=SimpleNamespace(model=model))
        ),
    )
    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(synchronize=lambda: None)
    )
    with pytest.raises(RuntimeError, match="interrupted weight transfer"):
        scheduler.sync_hf_weight(SyncHFWeightInput(weight_version=6))
    assert model.weight.item() == 1
    assert scheduler._rlinf_weight_version == 5
    with pytest.raises(RuntimeError, match="restart the rollout engine"):
        scheduler.resume_after_weight_update()
    with pytest.raises(RuntimeError, match="restart the rollout engine"):
        scheduler.sync_hf_weight(SyncHFWeightInput(weight_version=6))
