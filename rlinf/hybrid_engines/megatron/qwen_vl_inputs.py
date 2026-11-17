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

"""Verify canonical M-RoPE at the existing Bridge language-model boundary."""

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def verify_qwen_vl_positions(
    model: torch.nn.Module, expected: torch.Tensor, attention_mask: torch.Tensor
) -> Iterator[None]:
    """Check positions after Bridge's native THD packing, without overriding it."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs,
    )

    if not any(
        cls.__module__.startswith("megatron.bridge.models.qwen_vl.")
        for cls in type(model).__mro__
    ):
        raise ValueError(
            "Token-space position verification requires a Bridge Qwen-VL model"
        )
    checked = False

    def check_positions(module, args, kwargs):
        nonlocal checked
        actual = kwargs["position_ids"]
        canonical = expected
        if kwargs.get("packed_seq_params") is not None:
            canonical = (
                preprocess_packed_seqs(
                    canonical.permute(1, 2, 0),
                    attention_mask,
                    pre_process=True,
                    pg_collection=model.pg_collection,
                )[0]
                .permute(2, 0, 1)
                .contiguous()
            )
        if actual.shape != canonical.shape or not torch.equal(actual, canonical):
            raise ValueError(
                "Megatron-Bridge consumed M-RoPE positions different from SGLang"
            )
        checked = True

    handle = model.language_model.register_forward_pre_hook(
        check_positions, with_kwargs=True
    )
    try:
        yield
        if not checked:
            raise RuntimeError(
                "Bridge language forward did not expose M-RoPE positions"
            )
    finally:
        handle.remove()
