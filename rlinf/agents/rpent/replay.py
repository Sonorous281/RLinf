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

"""Read complete, hashed RPent episodes without sampling a policy or tools."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from omegaconf import DictConfig

from rlinf.agents.rpent.episode import episode_output, episode_task
from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker
from rlinf.config import shared_multi_modal_processor_enabled
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import AgentLoopOutput, MultiAgentLoopOutput


def _verified_json(path: Path, digest: str) -> dict:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        raise ValueError(
            f"Saved episode hash mismatch for {path}: {actual} != {digest}"
        )
    return json.loads(raw)


def _task_key(task: dict) -> str:
    return json.dumps(task, sort_keys=True, separators=(",", ":"))


class SavedRPentBatch:
    """Restore sampled turns from a verified complete-batch audit manifest.

    Request, outcome, turn and image files remain read-only. The reader rejects
    unavailable tasks instead of generating a replacement episode.
    """

    def __init__(
        self, source_dir: Path, manifest_path: Path, expected_version: int
    ) -> None:
        self.source_dir = source_dir.resolve()
        self.expected_version = expected_version
        self.manifest_path = manifest_path.resolve()
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        self.policy_files: dict[str, str] = manifest["policy_files"]
        if not self.policy_files:
            raise ValueError("Replay requires original-policy file hashes")
        if not manifest["passed"] or manifest["episodes"] != len(manifest["records"]):
            raise ValueError("Replay requires a passed manifest of complete episodes")
        if manifest["weight_versions"] != [expected_version]:
            raise ValueError(
                f"Replay requires only policy version {expected_version}; "
                f"found {manifest['weight_versions']}"
            )
        self._candidate_indexed = all(
            "candidate_id" in record for record in manifest["records"]
        )
        self._entries: dict[str | int, tuple[Path, dict, dict]] = {}
        self._verified_images: set[Path] = set()
        for record in manifest["records"]:
            directory = (self.source_dir / record["episode_id"]).resolve()
            if directory.parent != self.source_dir:
                raise ValueError(f"Episode directory must be inside {self.source_dir}")
            request = _verified_json(
                directory / "request.json", record["request_sha256"]
            )
            key = (
                record["candidate_id"]
                if self._candidate_indexed
                else _task_key(request["task"])
            )
            if key in self._entries:
                raise ValueError(f"Replay batch contains duplicate identity {key}")
            self._entries[key] = (directory, record, request["task"])

    def __len__(self) -> int:
        return len(self._entries)

    def contains_task(self, task: dict, candidate_id: int | None = None) -> bool:
        """Return whether this task belongs to the archived admitted batch."""
        key = candidate_id if self._candidate_indexed else _task_key(task)
        return key in self._entries and self._entries[key][2] == task

    def verify_policy(self, model_dir: Path) -> None:
        """Require the same policy, tokenizer and processor as the saved batch."""
        directory = model_dir.resolve()
        for name, expected in self.policy_files.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory):
                raise ValueError(f"Policy file must be inside {directory}: {name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Replay policy hash mismatch for {path}")

    def load_episode(
        self,
        task: dict,
        *,
        merge_prefix: bool,
        max_scored_tokens: int,
        candidate_id: int | None = None,
        token_space: bool = False,
    ) -> MultiAgentLoopOutput:
        """Read one original task's sampled IDs, logprobs and ordered image refs."""
        started = time.monotonic()
        key = candidate_id if self._candidate_indexed else _task_key(task)
        if key not in self._entries:
            raise ValueError(f"Task is absent from the saved replay batch: {key}")
        directory, record, original_task = self._entries[key]
        if original_task != task:
            raise ValueError(f"Saved candidate {candidate_id} has a different task")
        status = _verified_json(directory / "outcome.json", record["outcome_sha256"])
        if not status["terminal"] or status["success"] != record["success"]:
            raise ValueError(f"Expected a matching terminal outcome in {directory}")
        turns = []
        for index, item in enumerate(record["turns"]):
            name = f"turn-{index:05d}.json"
            if Path(item["file"]).name != name:
                raise ValueError(f"Expected chronological saved turn {name}")
            turn = _verified_json(directory / "training_audit" / name, item["sha256"])
            if turn["turn_id"] != index or turn["episode_id"] != record["episode_id"]:
                raise ValueError(f"Saved turn identity differs in {directory / name}")
            if turn["task"] != original_task:
                raise ValueError(
                    f"Saved turn has a different task in {directory / name}"
                )
            count = len(turn["output_ids"])
            if not count or any(
                len(turn[field]) != count
                for field in [
                    "response_mask",
                    "behavior_logprobs",
                    "token_weight_versions",
                ]
            ):
                raise ValueError(
                    f"Saved response fields have different lengths in {name}"
                )
            if any(x not in (0, 1) for x in turn["response_mask"]) or not any(
                turn["response_mask"]
            ):
                raise ValueError(
                    f"Saved response mask has no valid sampled tokens in {name}"
                )
            if not all(math.isfinite(x) for x in turn["behavior_logprobs"]):
                raise ValueError(f"Saved behavior logprobs are nonfinite in {name}")
            if (
                turn["weight_version_start"] != self.expected_version
                or turn["weight_version_end"] != self.expected_version
                or any(
                    x != self.expected_version for x in turn["token_weight_versions"]
                )
            ):
                raise ValueError(f"Saved turn has a different policy version in {name}")
            hashes = turn["image_sha256"]
            grid = turn["image_grid_thw"] or []
            if len(hashes) != len(grid):
                raise ValueError(f"Saved image order and token grids differ in {name}")
            refs = []
            for digest in hashes:
                if len(digest) != 64 or any(
                    x not in "0123456789abcdef" for x in digest
                ):
                    raise ValueError(f"Saved image has an invalid SHA256 in {name}")
                path = directory / "training_images" / digest
                if path not in self._verified_images:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual != digest:
                        raise ValueError(f"Saved image hash mismatch for {path}")
                    self._verified_images.add(path)
                refs.append({"path": str(path), "sha256": digest})
            multimodal = (
                {
                    "image_refs": refs,
                    "image_grid_thw": torch.tensor(grid, dtype=torch.long),
                    "num_images": len(refs),
                }
                if refs
                else None
            )
            if token_space and refs:
                contract = turn.get("multimodal_contract")
                if not contract or contract["image_grid_thw"] != grid:
                    raise ValueError(f"Saved multimodal contract differs in {name}")
                multimodal = copy.deepcopy(contract)
                if contract.get("format") == "sglang-token-space-v1":
                    if [ref["sha256"] for ref in contract["image_refs"]] != hashes:
                        raise ValueError(
                            f"Saved TokenSpace image order differs in {name}"
                        )
                elif contract.get("format") == "rlinf-qwen-image-inputs-v1":
                    from rlinf.hybrid_engines.sglang.multimodal.replay import (
                        replay_pixels,
                    )

                    if contract["prompt_token_ids"] != turn["input_ids"]:
                        raise ValueError(f"Saved serving prompt differs in {name}")
                    replay_pixels(multimodal)
                    for field in ("mrope_positions", "mrope_position_delta"):
                        multimodal[field] = torch.tensor(
                            multimodal[field], dtype=torch.long
                        )
                else:
                    raise ValueError(f"Unsupported saved multimodal format in {name}")
                multimodal["image_grid_thw"] = torch.tensor(grid, dtype=torch.long)
            turns.append(
                AgentLoopOutput(
                    prompt_ids=turn["input_ids"],
                    response_ids=turn["output_ids"],
                    response_mask=turn["response_mask"],
                    response_logprobs=turn["behavior_logprobs"],
                    response_weight_versions=turn["token_weight_versions"],
                    response_images=multimodal,
                    version_start=turn["weight_version_start"],
                    version_end=turn["weight_version_end"],
                    is_end=turn["finish_reason"] == "stop",
                    extra_fields={
                        "turn_id": turn["turn_id"],
                        "chain_id": turn["chain_id"],
                        "tool_names": turn["tool_names"],
                    },
                )
            )
        if not turns:
            raise ValueError(f"Saved episode has no sampled turns: {directory}")
        result = episode_output(
            turns,
            task=original_task,
            episode_id=record["episode_id"],
            status=status,
            elapsed_s=time.monotonic() - started,
            merge=merge_prefix,
            max_scored_tokens=max_scored_tokens,
        )
        result.extra_fields.update(infra_retries=0, infra_retry_errors=[])
        return result


class RPentReplayAgentLoopWorker(RPentAgentLoopWorker):
    """Supply saved complete episodes to the unchanged synchronous PPO runner."""

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        super().__init__(cfg, placement)
        self._saved_batch = SavedRPentBatch(
            Path(cfg.replay.source_dir),
            Path(cfg.replay.manifest),
            cfg.replay.expected_version,
        )
        if (
            len(self._saved_batch)
            != cfg.data.rollout_batch_size * cfg.algorithm.group_size
        ):
            raise ValueError("Saved replay batch size differs from rollout_batch_size")

    async def run_one_query(
        self, prompt_ids: list[int], answer: list[str]
    ) -> MultiAgentLoopOutput:
        """Load this task's archived result without invoking a model or tools."""
        candidate = self.rollout_candidate()
        return await asyncio.to_thread(
            self._saved_batch.load_episode,
            episode_task(answer),
            merge_prefix=self.cfg.agentloop.merge_prefix_turns,
            max_scored_tokens=self.cfg.agentloop.get(
                "max_scored_tokens", self.cfg.algorithm.sampling_params.max_new_tokens
            ),
            candidate_id=candidate["candidate_id"] if candidate else None,
            token_space=shared_multi_modal_processor_enabled(self.cfg.rollout.sglang),
        )


class RPentRecoveryAgentLoopWorker(RPentReplayAgentLoopWorker):
    """Recover one archived version-zero cohort, then collect online episodes.

    Unselected candidates wait for the existing cohort cancellation protocol.
    They neither regenerate partial trajectories nor enter the learner.
    """

    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        super().__init__(cfg, placement)
        manifest = json.loads(self._saved_batch.manifest_path.read_text())
        tasks = manifest["candidate_tasks"]
        group_size = cfg.algorithm.group_size
        expected = (
            cfg.data.get("sampling_batch_size", cfg.data.rollout_batch_size)
            * group_size
        )
        if len(tasks) != expected or any(
            task != tasks[index // group_size * group_size]
            for index, task in enumerate(tasks)
        ):
            raise ValueError("Recovery requires every original group and its members")
        self._recovery_tasks = tasks
        if sum(
            self._saved_batch.contains_task(task, index)
            for index, task in enumerate(tasks)
        ) != len(self._saved_batch):
            raise ValueError("Recovery candidates omit an admitted archived task")
        self._recovery_pending = True

    async def run_agentloop_rollout(self, input_channel, output_channel):
        """Consume the original candidate batch once before online collection."""
        try:
            return await super().run_agentloop_rollout(input_channel, output_channel)
        finally:
            self._recovery_pending = False

    async def run_one_query(self, prompt_ids: list[int], answer: list[str]):
        """Restore admitted data; leave discarded candidates for cancellation."""
        if not self._recovery_pending:
            return await RPentAgentLoopWorker.run_one_query(self, prompt_ids, answer)
        task = episode_task(answer)
        candidate = self.rollout_candidate()
        candidate_id = candidate["candidate_id"] if candidate else None
        if candidate_id is None or self._recovery_tasks[candidate_id] != task:
            raise ValueError(
                "Recovery input is absent from the original candidate cohort"
            )
        if self._saved_batch.contains_task(task, candidate_id):
            return await super().run_one_query(prompt_ids, answer)
        return await asyncio.Future()
