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

"""Preserve expanded Qwen image prompts through SGLang preprocessing."""

from collections.abc import Awaitable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

import torch
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
)
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor


@dataclass(kw_only=True)
class TokenizedImageInput(BaseMultiModalProcessorOutput):
    """Loaded images with the caller's already expanded prompt IDs."""

    input_ids: list[int]


class TokenizedQwenVLProcessor(QwenVLImageProcessor):
    """Keep sampled token boundaries when processing Qwen3.5 image requests.

    Token-ID requests must contain expanded image pads and raw image data.
    Text requests retain SGLang's normal processor behavior. Register through
    ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE``; the shared installation is untouched.
    """

    models = [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration]

    def load_mm_data(
        self, prompt: str | list[int], **kwargs: Any
    ) -> BaseMultiModalProcessorOutput | Awaitable[BaseMultiModalProcessorOutput]:
        """Carry token IDs alongside the upstream image-loading result."""
        if not isinstance(prompt, list):
            return super().load_mm_data(prompt=prompt, **kwargs)
        if (
            kwargs.get("video_data")
            or kwargs.get("audio_data")
            or any(isinstance(image, dict) for image in kwargs.get("image_data") or [])
        ):
            raise ValueError(
                "Token-preserving Qwen prompts require raw images; "
                "video, audio and precomputed image inputs are unsupported."
            )
        image_text, image_count = self.mm_tokens.image_token_regex.subn(
            self.mm_tokens.image_token, self._tokenizer.decode(prompt)
        )
        if image_count != len(kwargs.get("image_data") or []):
            raise ValueError(
                f"Prompt contains {image_count} image spans, but received "
                f"{len(kwargs.get('image_data') or [])} images."
            )
        loaded = super().load_mm_data(prompt=image_text, **kwargs)

        def with_prompt_ids(result):
            return TokenizedImageInput(
                input_text=result.input_text,
                images=result.images,
                videos=result.videos,
                audios=result.audios,
                input_ids=prompt,
            )

        # SGLang's asynchronous loader must finish before its fields are read.
        if isawaitable(loaded):

            async def load_with_prompt_ids():
                return with_prompt_ids(await loaded)

            return load_with_prompt_ids()
        return with_prompt_ids(loaded)

    def process_and_combine_mm_data(
        self, base_output: BaseMultiModalProcessorOutput, mm_tokens: Any, **kwargs: Any
    ) -> tuple[list[Any], torch.Tensor, Any]:
        """Pass request-local IDs to image processing before offsets are built."""
        if isinstance(base_output, TokenizedImageInput):
            kwargs["prompt_ids"] = base_output.input_ids
        return super().process_and_combine_mm_data(base_output, mm_tokens, **kwargs)

    def process_mm_data(
        self, input_text: str, prompt_ids: list[int] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Use processor pixels while retaining the caller's text/action tokens."""
        processed = super().process_mm_data(input_text=input_text, **kwargs)
        if prompt_ids is None:
            return processed
        input_ids = torch.tensor(prompt_ids, dtype=torch.long)
        offsets = self.get_mm_items_offset(input_ids, self.IM_TOKEN_ID)
        actual = [end - start + 1 for start, end in offsets]
        expected = (
            processed["image_grid_thw"].prod(dim=-1) // self.spatial_merge_size**2
        ).tolist()
        if actual != expected or any(
            start == 0
            or end + 1 == len(prompt_ids)
            or prompt_ids[start - 1] != self.vision_start_token_id
            or prompt_ids[end + 1] != self.vision_end_token_id
            for start, end in offsets
        ):
            raise ValueError(
                "Expected expanded image pads enclosed by vision markers: "
                f"processor grid needs {expected}, prompt contains {actual}."
            )
        # Re-tokenizing decoded sampled text can change even equal-length IDs.
        # Offsets and M-RoPE must be derived from this exact sequence instead.
        processed["input_ids"] = input_ids.unsqueeze(0)
        if "attention_mask" in processed:
            processed["attention_mask"] = torch.ones_like(processed["input_ids"])
        if "mm_token_type_ids" in processed:
            processed["mm_token_type_ids"] = (
                processed["input_ids"] == self.IM_TOKEN_ID
            ).long()
        return processed
