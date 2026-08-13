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
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Optional, TypedDict, Union

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.robotwin.seed_utils import partition_success_seeds
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboTwinEnv"]


class ExactSeedMismatch(RuntimeError):
    """Raised when RoboTwin initializes a seed other than the requested seed."""

    def __init__(self, requested_seed: int, actual_seed: int):
        super().__init__(
            "RoboTwin exact seed mismatch: "
            f"requested {requested_seed}, initialized {actual_seed}"
        )
        self.requested_seed = requested_seed
        self.actual_seed = actual_seed


class RoboTwinNativeStatus(TypedDict):
    """Facts read directly from the active RoboTwin task."""

    eval_success: bool
    take_action_cnt: int
    step_lim: int | None
    actual_seed: int


class RoboTwinEpisodeStatus(RoboTwinNativeStatus):
    """Native status plus RLinf's agent-episode validity."""

    agent_valid: bool
    invalid_reason: str | None


class RoboTwinExecutionResult(TypedDict):
    """Result of an agent action request."""

    action_type: Literal["qpos", "ee"]
    requested_actions: int
    executed_actions: int
    episode_status: RoboTwinEpisodeStatus


class RoboTwinResetResult(TypedDict):
    """Result of an exact-seed reset."""

    requested_seed: int
    actual_seed: int
    instruction: str
    episode_status: RoboTwinEpisodeStatus


def _camera_geometry(camera: Any) -> tuple[np.ndarray, np.ndarray]:
    """Derive metric depth and world xyz from one Position texture read."""
    position = camera.get_picture("Position")
    invalid = position[..., 3] >= 1

    depth = (-position[..., 2]).astype(np.float32)
    depth[invalid] = np.nan

    model = np.asarray(camera.get_model_matrix())
    world = position[..., :3] @ model[:3, :3].T + model[:3, 3]
    world = world.astype(np.float32)
    world[invalid] = np.nan
    return depth, world


def _camera_meta(camera: Any) -> dict[str, Any]:
    """Return calibration metadata from a native RoboTwin camera."""
    return {
        "intrinsic_K": np.asarray(camera.get_intrinsic_matrix()),
        "extrinsic_cv": np.asarray(camera.get_extrinsic_matrix()),
        "cam2world_gl": np.asarray(camera.get_model_matrix()),
        "width": int(camera.get_width()),
        "height": int(camera.get_height()),
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
        self._invalid_agent_envs: set[int] = set()
        self._invalid_agent_reasons: dict[int, str] = {}

        self._init_env()

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
        os.environ["ROBOTWIN_ASSETS_ROOT"] = self.cfg.assets_path

        from robotwin.envs.vector_env import VectorEnv

        initial_env_seeds = self.cfg.get("initial_env_seeds", None)
        if initial_env_seeds is None:
            env_seeds = self.reset_state_ids.tolist()
        else:
            env_seeds = [int(seed) for seed in initial_env_seeds]
            if len(env_seeds) != self.num_envs:
                raise ValueError(
                    "initial_env_seeds must contain one seed per environment"
                )

        self.venv = VectorEnv(
            task_config=OmegaConf.to_container(self.cfg.task_config, resolve=True),
            n_envs=self.num_envs,
            env_seeds=env_seeds,
        )

    def _sub_env(self, env_id: int):
        envs = getattr(self.venv, "envs", None)
        if not isinstance(envs, list) or not 0 <= env_id < len(envs):
            raise IndexError(f"RoboTwin env_id {env_id} is unavailable")
        return envs[env_id]

    @staticmethod
    def _task_seed(sub_env: Any) -> int:
        value = getattr(sub_env.task, "ep_num", None)
        if value is None:
            value = getattr(sub_env, "env_seed", None)
        if value is None:
            raise RuntimeError("native RoboTwin episode seed is unavailable")
        return int(value)

    def _native_status_unlocked(self, sub_env: Any) -> RoboTwinNativeStatus:
        task = sub_env.task
        step_limit = getattr(task, "step_lim", None)
        if step_limit is None:
            task_args = getattr(sub_env, "args", None)
            if isinstance(task_args, dict):
                step_limit = task_args.get("step_lim")
            elif task_args is not None:
                step_limit = getattr(task_args, "step_lim", None)
        if step_limit is None:
            cfg_get = getattr(self.cfg, "get", None)
            step_limit = (
                cfg_get("max_episode_steps", None)
                if callable(cfg_get)
                else getattr(self.cfg, "max_episode_steps", None)
            )
        return {
            "eval_success": bool(getattr(task, "eval_success", False)),
            "take_action_cnt": int(getattr(task, "take_action_cnt", 0)),
            "step_lim": int(step_limit) if step_limit is not None else None,
            "actual_seed": self._task_seed(sub_env),
        }

    def _episode_status_unlocked(
        self, env_id: int, sub_env: Any
    ) -> RoboTwinEpisodeStatus:
        return {
            **self._native_status_unlocked(sub_env),
            "agent_valid": env_id not in self._invalid_agent_envs,
            "invalid_reason": self._invalid_agent_reasons.get(env_id),
        }

    @staticmethod
    def _execution_should_stop(status: RoboTwinNativeStatus) -> bool:
        step_limit = status["step_lim"]
        return status["eval_success"] or (
            step_limit is not None and status["take_action_cnt"] >= step_limit
        )

    def _require_agent_valid(self, env_id: int) -> None:
        if env_id in self._invalid_agent_envs:
            reason = self._invalid_agent_reasons.get(env_id, "invalid_agent_episode")
            raise RuntimeError(
                f"RoboTwin agent environment {env_id} is invalid: {reason}; "
                "reset_exact() is required"
            )

    def _mark_agent_invalid(self, env_id: int, reason: str) -> None:
        self._invalid_agent_envs.add(env_id)
        self._invalid_agent_reasons[env_id] = reason

    def _clear_agent_invalid(self, env_id: int) -> None:
        self._invalid_agent_envs.discard(env_id)
        self._invalid_agent_reasons.pop(env_id, None)

    @contextmanager
    def _native_reset_args(self, sub_env: Any) -> Iterator[None]:
        # Native RoboTwin reset requires step_lim in SubEnv.args.
        args = getattr(sub_env, "args", None)
        if not isinstance(args, dict):
            raise TypeError("native RoboTwin SubEnv.args must be a dict")
        if args.get("step_lim") is not None:
            yield
            return

        step_limit = getattr(getattr(sub_env, "task", None), "step_lim", None)
        if step_limit is None:
            step_limit = self.cfg.get("max_episode_steps", None)
        if step_limit is None:
            raise RuntimeError("native RoboTwin reset step limit is unavailable")

        sub_env.args = {**args, "step_lim": int(step_limit)}
        try:
            yield
        finally:
            sub_env.args = args

    @staticmethod
    def _robot_state_unlocked(sub_env: Any) -> dict[str, Any]:
        robot = sub_env.task.robot
        left_target = robot.get_left_arm_jointState()
        right_target = robot.get_right_arm_jointState()
        left_real = robot.get_left_arm_real_jointState()
        right_real = robot.get_right_arm_real_jointState()
        return {
            "left_eef_pose": np.asarray(robot.get_left_ee_pose(), dtype=np.float64),
            "right_eef_pose": np.asarray(robot.get_right_ee_pose(), dtype=np.float64),
            "left_tcp_pose": np.asarray(robot.get_left_tcp_pose(), dtype=np.float64),
            "right_tcp_pose": np.asarray(robot.get_right_tcp_pose(), dtype=np.float64),
            "left_gripper": float(robot.get_left_gripper_val()),
            "right_gripper": float(robot.get_right_gripper_val()),
            "qpos_target14": np.asarray(left_target + right_target, dtype=np.float64),
            "arm_qpos_real12": np.asarray(
                left_real[:-1] + right_real[:-1], dtype=np.float64
            ),
        }

    @staticmethod
    def _native_cameras_unlocked(sub_env: Any) -> dict[str, Any]:
        cameras = sub_env.task.cameras
        result: dict[str, Any] = {}
        for camera, name in zip(cameras.static_camera_list, cameras.static_camera_name):
            if name == "head_camera":
                result["head"] = camera
                break
        if "head" not in result:
            raise RuntimeError("native RoboTwin head camera is unavailable")
        if bool(getattr(cameras, "collect_wrist_camera", False)):
            result["left_wrist"] = cameras.left_camera
            result["right_wrist"] = cameras.right_camera
        return result

    def execute_action_chunk(
        self,
        actions: Any,
        *,
        action_type: Literal["qpos", "ee"],
        env_id: int = 0,
    ) -> RoboTwinExecutionResult:
        """Execute native qpos14 or eef16 actions outside training rollouts."""
        self._require_agent_valid(env_id)
        if action_type not in ("qpos", "ee"):
            raise ValueError("action_type must be 'qpos' or 'ee'")
        array = np.asarray(actions, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        expected_dim = 14 if action_type == "qpos" else 16
        if array.ndim != 2 or array.shape[1] != expected_dim:
            raise ValueError(
                f"{action_type} actions must have shape [N,{expected_dim}]"
            )

        sub_env = self._sub_env(env_id)
        executed = 0
        with sub_env.lock:
            for action in array:
                status = self._native_status_unlocked(sub_env)
                if self._execution_should_stop(status):
                    break
                before = status["take_action_cnt"]
                sub_env.task.take_action(action, action_type=action_type)
                after = int(getattr(sub_env.task, "take_action_cnt", 0))
                if after > before:
                    executed += 1
            episode_status = self._episode_status_unlocked(env_id, sub_env)
        return {
            "action_type": action_type,
            "requested_actions": int(len(array)),
            "executed_actions": executed,
            "episode_status": episode_status,
        }

    def apply_qpos_updates(
        self, updates: list[dict[str, Any]], *, env_id: int = 0
    ) -> RoboTwinExecutionResult:
        """Apply sparse arm/gripper updates against freshly read qpos14 state."""
        self._require_agent_valid(env_id)
        normalized: list[dict[str, Any]] = []
        for update in updates:
            arm = update.get("arm")
            if arm not in ("left", "right"):
                raise ValueError("arm must be 'left' or 'right'")
            item: dict[str, Any] = {"arm": arm}
            if update.get("arm_qpos") is not None:
                arm_qpos = np.asarray(update["arm_qpos"], dtype=np.float64)
                if arm_qpos.shape != (6,):
                    raise ValueError("arm_qpos must have shape (6,)")
                item["arm_qpos"] = arm_qpos
            if update.get("gripper") is not None:
                item["gripper"] = float(update["gripper"])
            if len(item) == 1:
                raise ValueError("qpos update must set arm_qpos and/or gripper")
            normalized.append(item)

        sub_env = self._sub_env(env_id)
        executed = 0
        with sub_env.lock:
            for update in normalized:
                status = self._native_status_unlocked(sub_env)
                if self._execution_should_stop(status):
                    break
                qpos_target14 = self._robot_state_unlocked(sub_env)[
                    "qpos_target14"
                ].copy()
                offset = 0 if update["arm"] == "left" else 7
                if "arm_qpos" in update:
                    qpos_target14[offset : offset + 6] = update["arm_qpos"]
                if "gripper" in update:
                    qpos_target14[offset + 6] = update["gripper"]
                before = status["take_action_cnt"]
                sub_env.task.take_action(qpos_target14, action_type="qpos")
                after = int(getattr(sub_env.task, "take_action_cnt", 0))
                if after > before:
                    executed += 1
            episode_status = self._episode_status_unlocked(env_id, sub_env)
        return {
            "action_type": "qpos",
            "requested_actions": len(normalized),
            "executed_actions": executed,
            "episode_status": episode_status,
        }

    def capture_observation(self, env_id: int = 0) -> dict[str, Any]:
        """Capture one observation while holding the native environment lock."""
        self._require_agent_valid(env_id)
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            native_observation = sub_env.task.get_obs()["observation"]
            camera_keys = {
                "head": "head_camera",
                "left_wrist": "left_camera",
                "right_wrist": "right_camera",
            }
            views = {}
            for name, camera in self._native_cameras_unlocked(sub_env).items():
                depth, world_xyz = _camera_geometry(camera)
                views[name] = {
                    "rgb": native_observation[camera_keys[name]]["rgb"],
                    "depth": depth,
                    "world_xyz": world_xyz,
                    "camera_meta": _camera_meta(camera),
                }
            object_names = []
            for entity in sub_env.task.scene.get_all_actors():
                name = entity.get_name()
                if name and name not in ("table", "wall", "ground"):
                    object_names.append(name)
            return {
                "views": views,
                "robot_state": self._robot_state_unlocked(sub_env),
                "task_name": sub_env.task_name,
                "task_language": sub_env.task.get_instruction(),
                "object_names": object_names,
                "depth_unit": "metres",
                "world_frame": "world",
            }

    def get_robot_state(self, env_id: int = 0) -> dict[str, Any]:
        """Return target qpos, measured arm qpos, poses, and grippers."""
        self._require_agent_valid(env_id)
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            return self._robot_state_unlocked(sub_env)

    def get_episode_status(self, env_id: int = 0) -> dict[str, Any]:
        """Return native status plus RLinf's agent-episode validity."""
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            return self._episode_status_unlocked(env_id, sub_env)

    def plan_arm_path(
        self,
        env_id: int,
        arm: Literal["left", "right"],
        target_pose: Any,
    ) -> dict[str, Any]:
        """Forward one pose-planning query to RoboTwin's native planner."""
        self._require_agent_valid(env_id)
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError("target_pose must have shape (7,)")
        sub_env = self._sub_env(env_id)
        with sub_env.lock:
            planner = (
                sub_env.task.robot.left_plan_path
                if arm == "left"
                else sub_env.task.robot.right_plan_path
            )
            result = planner(target.tolist())
            return {
                "status": result.get("status", "Unknown"),
                "position": (
                    np.asarray(result["position"], dtype=np.float64)
                    if result.get("position") is not None
                    else None
                ),
                "velocity": (
                    np.asarray(result["velocity"], dtype=np.float64)
                    if result.get("velocity") is not None
                    else None
                ),
            }

    def reset_exact(self, env_id: int, seed: int) -> RoboTwinResetResult:
        """Reset an environment and require the requested seed."""
        sub_env = self._sub_env(env_id)
        requested_seed = int(seed)
        self._mark_agent_invalid(env_id, "reset_in_progress")
        try:
            with self._native_reset_args(sub_env):
                self.venv.reset(env_idx=[env_id], env_seeds=[requested_seed])
            sub_env = self._sub_env(env_id)
            with sub_env.lock:
                actual_seed = self._task_seed(sub_env)
                instruction = sub_env.task.get_instruction()
            if actual_seed != requested_seed:
                self._mark_agent_invalid(env_id, "exact_seed_mismatch")
                raise ExactSeedMismatch(requested_seed, actual_seed)
            self.reset_state_ids[env_id] = actual_seed
            self._is_start = False
            self._reset_metrics([env_id])
            with sub_env.lock:
                native_status = self._native_status_unlocked(sub_env)
            result: RoboTwinResetResult = {
                "requested_seed": requested_seed,
                "actual_seed": actual_seed,
                "instruction": instruction,
                "episode_status": {
                    **native_status,
                    "agent_valid": True,
                    "invalid_reason": None,
                },
            }
        except ExactSeedMismatch:
            raise
        except Exception:
            self._mark_agent_invalid(env_id, "exact_seed_reset_failed")
            raise
        else:
            self._clear_agent_invalid(env_id)
            return result

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
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
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
