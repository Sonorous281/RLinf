# Copyright 2025 The RLinf Authors.
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

import json
import os
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.robotwin.seed_utils import partition_success_seeds
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboTwinEnv"]

ROBOTWIN_HYBRID_CONTRACT_VERSION = "robotwin-agent-v1"
ROBOTWIN_COMPATIBILITY_ID = "robotwin-rpent-downloads-2026-07-31-v1"
ROBOTWIN_ASSET_FILE_COUNT = 20854
ROBOTWIN_ASSET_TREE_SHA256 = (
    "6ef76bdd0b4b8fefbc5d8dc855563e3b4c4c03674c586079141ab2b66079c12b"
)
ROBOTWIN_CUROBO_REPOSITORY = "https://github.com/NVlabs/curobo.git"
ROBOTWIN_CUROBO_REVISION = "2fbffc35225398cf9d5f382804faa9de2608753b"
LINGBOT_CHECKPOINT = "RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500"
LINGBOT_CHECKPOINT_REVISION = "c55199f25a10397e79dce177ee11c8774fb8edde"
ROBOTWIN_REQUIRED_CAPABILITIES = {
    "robot_state",
    "policy_observation",
    "agent_observation",
    "debug_state",
    "episode_status",
    "plan_arm_path",
    "execute_actions",
    "execute_qpos_updates",
    "reset_episode",
    "mutation_result",
}


class RoboTwinEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        self.profile = cfg.get("profile", "standard")
        self.allow_hybrid_debug = bool(cfg.get("allow_hybrid_debug", False))
        self._validate_profile_config(self.profile, num_envs, cfg.auto_reset)

        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.base_seed = env_seed
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations

        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = cfg.use_custom_reward

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True

        self.task_name = cfg.task_config.task_name

        self.center_crop = cfg.get("center_crop", False)
        self._init_reset_state_ids()

        self._init_env()
        if self.profile == "downloads_hybrid":
            self.get_capabilities()

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()
            self._elapsed_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _init_env(self):
        mp.set_start_method("spawn", force=True)
        os.environ["ASSETS_PATH"] = self.cfg.assets_path

        from robotwin.envs.vector_env import VectorEnv

        if self.profile == "downloads_hybrid":
            env_seeds = [int(self.cfg.get("hybrid_initial_seed", self.seed))]
        else:
            env_seeds = self.reset_state_ids.tolist()

        vector_env_kwargs = {
            "task_config": OmegaConf.to_container(self.cfg.task_config, resolve=True),
            "n_envs": self.num_envs,
            "env_seeds": env_seeds,
        }
        if self.profile == "downloads_hybrid":
            vector_env_kwargs["profile"] = self.profile

        self.venv = VectorEnv(
            **vector_env_kwargs,
        )

    @staticmethod
    def _validate_profile_config(profile: str, num_envs: int, auto_reset: bool) -> None:
        """Validate profile invariants before constructing the native simulator."""
        if profile not in ("standard", "downloads_hybrid"):
            raise ValueError(
                "RoboTwin profile must be 'standard' or 'downloads_hybrid'"
            )
        if profile == "downloads_hybrid" and num_envs != 1:
            raise ValueError("downloads_hybrid requires num_envs == 1")
        if profile == "downloads_hybrid" and auto_reset:
            raise ValueError("downloads_hybrid requires auto_reset == False")

    def _require_hybrid_profile(self) -> None:
        if self.profile != "downloads_hybrid":
            raise RuntimeError(
                "RoboTwin native capabilities require profile='downloads_hybrid'"
            )

    def _call_hybrid_capability(self, method: str, *args, **kwargs):
        self._require_hybrid_profile()
        capability = getattr(self.venv, method, None)
        if capability is None:
            raise RuntimeError(
                f"RoboTwin native compatibility branch is missing {method}()"
            )
        return capability(*args, **kwargs)

    def get_capabilities(self) -> dict[str, Any]:
        """Return and validate the RoboTwin hybrid capability contract."""
        native = self._call_hybrid_capability("get_capabilities")
        compatibility_id = native.get("compatibility_id")
        if compatibility_id != ROBOTWIN_COMPATIBILITY_ID:
            raise RuntimeError(
                "RoboTwin compatibility mismatch: "
                f"expected {ROBOTWIN_COMPATIBILITY_ID!r}, "
                f"got {compatibility_id!r}"
            )
        expected_asset_snapshot = {
            "file_count": ROBOTWIN_ASSET_FILE_COUNT,
            "tree_sha256": ROBOTWIN_ASSET_TREE_SHA256,
        }
        if native.get("asset_snapshot") != expected_asset_snapshot:
            raise RuntimeError(
                "RoboTwin asset snapshot contract mismatch: "
                f"expected {expected_asset_snapshot!r}, "
                f"got {native.get('asset_snapshot')!r}"
            )
        native_capabilities = set(native.get("capabilities", ()))
        missing = sorted(ROBOTWIN_REQUIRED_CAPABILITIES - native_capabilities)
        if missing:
            raise RuntimeError(
                f"RoboTwin compatibility branch is missing capabilities: {missing}"
            )
        planner_snapshot = native.get("planner_snapshot", {})
        expected_planner = {
            "repository": ROBOTWIN_CUROBO_REPOSITORY,
            "revision": ROBOTWIN_CUROBO_REVISION,
            "clean": True,
        }
        for key, expected in expected_planner.items():
            if planner_snapshot.get(key) != expected:
                raise RuntimeError(
                    "RoboTwin planner snapshot contract mismatch: "
                    f"expected {key}={expected!r}, "
                    f"got {planner_snapshot.get(key)!r}"
                )
        server_instance_id = native.get("server_instance_id")
        mutation_id_prefix = native.get("mutation_id_prefix")
        if (
            not isinstance(server_instance_id, str)
            or not server_instance_id
            or mutation_id_prefix != f"{server_instance_id}:"
        ):
            raise RuntimeError(
                "RoboTwin compatibility branch returned an invalid mutation scope"
            )
        return {
            **native,
            "contract_version": ROBOTWIN_HYBRID_CONTRACT_VERSION,
            "profile": self.profile,
            "action_specs": {
                "qpos": {
                    "layout": "qpos14",
                    "shape": [14],
                    "dtype": "float64",
                },
                "ee": {
                    "layout": "eef16",
                    "shape": [16],
                    "dtype": "float64",
                    "frame": "world",
                    "position_unit": "metres",
                    "quaternion_order": "wxyz",
                    "fields": [
                        "left_xyz",
                        "left_qwxyz",
                        "left_gripper",
                        "right_xyz",
                        "right_qwxyz",
                        "right_gripper",
                    ],
                },
            },
            "camera_specs": {
                "policy": {
                    "views": [
                        "cam_high",
                        "cam_left_wrist",
                        "cam_right_wrist",
                    ],
                    "native_resolution": [320, 240],
                    "model_resolution": [224, 224],
                },
                "agent": {
                    "native_views": ["head", "left_wrist", "right_wrist"],
                    "high_resolution": [1024, 1024],
                    "depth_unit": "metres",
                    "world_frame": "world",
                },
            },
            "episode_budget_counter": "take_action_cnt",
            "canonical_success": "TASK_ENV.eval_success",
            "planner_contract": {
                **expected_planner,
                "backend": "curobo",
            },
            "mutation_protocol": {
                "scope": "server_instance",
                "guarantee": "at-most-once",
                "query_method": "get_mutation_result",
            },
            "model_contract": {
                "checkpoint": LINGBOT_CHECKPOINT,
                "revision": LINGBOT_CHECKPOINT_REVISION,
                "policy_name": "robotwin_eef",
                "norm_stats": "norm_stats/robotwin_eef.json",
                "qwen_base": "qwen_base",
                "camera_order": [
                    "cam_high",
                    "cam_left_wrist",
                    "cam_right_wrist",
                ],
                "state_layout": "eef16",
                "action_layout": "eef16",
                "default_use_length": 50,
            },
        }

    def get_robot_state(self, env_id: int = 0) -> dict[str, Any]:
        """Return dual-arm EEF, gripper, and qpos state."""
        return self._call_hybrid_capability("get_robot_state", env_id=env_id)

    def capture_policy_observation(self, env_id: int = 0) -> dict[str, Any]:
        """Capture the native three-camera LingBot observation."""
        return self._call_hybrid_capability("capture_policy_observation", env_id=env_id)

    def capture_agent_observation(self, env_id: int = 0) -> dict[str, Any]:
        """Capture synchronized agent-visible RGB and geometry."""
        return self._call_hybrid_capability("capture_agent_observation", env_id=env_id)

    def capture_debug_state(self, env_id: int = 0) -> dict[str, Any]:
        """Capture simulator oracle state for tests and evaluators only."""
        if not self.allow_hybrid_debug:
            raise PermissionError("capture_debug_state is disabled by configuration")
        return self._call_hybrid_capability("capture_debug_state", env_id=env_id)

    def get_episode_status(self, env_id: int = 0) -> dict[str, Any]:
        """Return canonical native success and budget status."""
        return self._call_hybrid_capability("get_episode_status", env_id=env_id)

    def plan_arm_path(
        self, env_id: int, arm: str, target_pose: np.ndarray
    ) -> dict[str, Any]:
        """Plan one native arm path without mutating the environment."""
        return self._call_hybrid_capability("plan_arm_path", env_id, arm, target_pose)

    def execute_actions(
        self,
        env_id: int,
        action_type: str,
        actions: np.ndarray,
        mutation_id: str,
        expected_version: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        """Execute qpos14 or eef16 actions through the native task owner."""
        return self._call_hybrid_capability(
            "execute_actions",
            env_id,
            action_type,
            actions,
            mutation_id,
            expected_version,
        )

    def execute_qpos_updates(
        self,
        env_id: int,
        updates: list[dict[str, Any]],
        mutation_id: str,
        expected_plan: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        """Execute per-waypoint qpos updates using freshly read native state."""
        return self._call_hybrid_capability(
            "execute_qpos_updates",
            env_id,
            updates,
            mutation_id,
            expected_plan,
        )

    def reset_episode(
        self,
        env_id: int,
        seed: int,
        mutation_id: str,
        reset_options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Reset through the Downloads-compatible native lifecycle."""
        return self._call_hybrid_capability(
            "reset_episode",
            env_id,
            seed,
            mutation_id,
            reset_options,
        )

    def get_mutation_result(
        self, env_id: int, mutation_id: str
    ) -> Optional[dict[str, Any]]:
        """Return the cached state of a native mutation."""
        return self._call_hybrid_capability("get_mutation_result", env_id, mutation_id)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            if isinstance(infos["success"], list):
                infos["success"] = torch.as_tensor(
                    np.array(infos["success"]).reshape(-1), device=self.device
                )
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def center_and_crop(self, image, center_crop=False):
        image = np.array(image)

        image = Image.fromarray(image).convert("RGB")
        if center_crop:
            image = center_crop_image(image)
        return np.array(image)

    def _extract_obs_image(self, raw_obs):
        batch_images = []
        batch_wrist_images = []
        batch_states = []
        batch_instructions = []
        for obs in raw_obs:
            batch_images.append(
                self.center_and_crop(obs["full_image"], center_crop=self.center_crop)
            )
            wrist_images = []
            if "left_wrist_image" in obs and obs["left_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["left_wrist_image"], center_crop=self.center_crop
                    )
                )
            if "right_wrist_image" in obs and obs["right_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["right_wrist_image"], center_crop=self.center_crop
                    )
                )
            if len(wrist_images) > 0:
                batch_wrist_images.append(
                    torch.stack([torch.from_numpy(img) for img in wrist_images])
                )
            batch_states.append(obs["state"])
            batch_instructions.append(obs["instruction"])

        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        if len(batch_wrist_images) > 0:
            batch_wrist_images = torch.stack(batch_wrist_images)
        else:
            batch_wrist_images = None
        batch_states = torch.stack([torch.from_numpy(state) for state in batch_states])

        extracted_obs = {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
        }

        return extracted_obs

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _cal_chunk_rewards(self, step_reward, chunk_step, terminations, infos):
        n_steps_to_run = np.array(
            [[0] for i in range(self.num_envs)]
        )  # infos.get("n_steps_to_run", np.array([[0] for i in range(self.num_envs)]))

        n_steps_to_run = torch.as_tensor(
            np.array(n_steps_to_run).reshape(-1), device=self.device
        )
        chunk_rewards = torch.zeros(self.num_envs, chunk_step, device=self.device)
        for env_id in range(self.num_envs):
            steps_left = n_steps_to_run[env_id]
            reward = step_reward[env_id]
            start_idx = chunk_step - steps_left - 1

            if terminations[env_id] and start_idx > 0:
                if self.use_rel_reward:
                    chunk_rewards[env_id, start_idx] = reward
                else:
                    chunk_rewards[env_id, start_idx:] = reward

        return chunk_rewards

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ):
        if self._is_start:
            self._is_start = False

        env_seeds = self.reset_state_ids.tolist() if env_seeds is None else env_seeds

        self.venv.reset(env_idx=env_idx, env_seeds=env_seeds)
        raw_obs = self.venv.get_obs()
        infos = {}

        self._reset_metrics(env_idx)

        extracted_obs = self._extract_obs_image(raw_obs)

        return extracted_obs, infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, dict] = None, auto_reset=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)

        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        self._elapsed_steps += actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        # chunk_actions: [num_envs, chunk_step, action_dim]
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            chunk_actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        obs_list.append(extracted_obs)
        infos_list.append(infos)
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        chunk_rewards = self._cal_chunk_rewards(
            step_reward, chunk_step, terminations, infos
        )

        self._elapsed_steps += chunk_actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        past_dones = torch.logical_or(terminations, truncations)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_terminations[:, -1] = terminations

        chunk_truncations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_truncations[:, -1] = truncations

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        final_info = infos.copy()
        if self.cfg.is_eval:
            self.update_reset_state_ids(env_idx=env_idx)

        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        # Gymnasium calls it final observation, but this is the actual
        # o_{t+1} / true next observation.
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def offload(self, clear_cache=True):
        if hasattr(self, "venv"):
            self.venv.close(clear_cache)

    def sample_action_space(self):
        return np.random.randn(self.num_envs, self.horizon, 14)

    def _init_reset_state_ids(self):
        if self.cfg.get("seeds_path", None) is not None and os.path.exists(
            self.cfg.seeds_path
        ):
            with open(self.cfg.seeds_path, "r") as f:
                data = json.load(f)
            success_seeds = data[self.task_name].get("success_seeds", None)
            if success_seeds is not None:
                success_seeds = torch.as_tensor(success_seeds, dtype=torch.long)
                self.success_seeds = partition_success_seeds(
                    success_seeds,
                    base_seed=self.base_seed,
                    seed_offset=self.seed_offset,
                    total_num_processes=self.total_num_processes,
                    num_group=self.num_group,
                )
                self._current_seed_index = 0
            else:
                self.success_seeds = None
                self._current_seed_index = 0
        else:
            self.success_seeds = None
            self._current_seed_index = 0

        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None):
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return

        if env_idx is not None and hasattr(self, "reset_state_ids"):
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            for idx in env_idx:
                self.reset_state_ids[idx] = reset_state_ids[idx]
        else:
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            self.reset_state_ids = reset_state_ids

    def check_seeds(self, seeds):
        resutls = self.venv.check_seeds(seeds)

        return resutls
