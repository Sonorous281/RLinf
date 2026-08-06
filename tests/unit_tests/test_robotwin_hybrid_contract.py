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

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.robotwin import robotwin_env as robotwin_env_module
from rlinf.envs.robotwin.robotwin_env import (
    ROBOTWIN_COMPATIBILITY_ID,
    RoboTwinEnv,
)


class FakeVectorEnv:
    def __init__(self, compatibility_id=ROBOTWIN_COMPATIBILITY_ID):
        self.compatibility_id = compatibility_id
        self.calls = []

    def get_capabilities(self):
        return {
            "compatibility_id": self.compatibility_id,
            "asset_snapshot": {
                "file_count": 20854,
                "tree_sha256": (
                    "6ef76bdd0b4b8fefbc5d8dc855563e3b4c4c03674c586079141ab2b66079c12b"
                ),
            },
            "planner_snapshot": {
                "repository": robotwin_env_module.ROBOTWIN_CUROBO_REPOSITORY,
                "revision": robotwin_env_module.ROBOTWIN_CUROBO_REVISION,
                "source_root": "/opt/curobo/src",
                "clean": True,
            },
            "server_instance_id": "server-1",
            "mutation_id_prefix": "server-1:",
            "capabilities": [
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
            ],
        }

    def get_robot_state(self, env_id=0):
        self.calls.append(("get_robot_state", env_id))
        return {"qpos14": np.zeros(14)}

    def capture_policy_observation(self, env_id=0):
        self.calls.append(("capture_policy_observation", env_id))
        return {"task": "place the fan"}

    def capture_agent_observation(self, env_id=0):
        self.calls.append(("capture_agent_observation", env_id))
        return {"frame_id": 1}

    def capture_debug_state(self, env_id=0):
        self.calls.append(("capture_debug_state", env_id))
        return {"actors": {}}

    def get_episode_status(self, env_id=0):
        self.calls.append(("get_episode_status", env_id))
        return {"success": False}

    def plan_arm_path(self, env_id, arm, target_pose):
        self.calls.append(("plan_arm_path", env_id, arm, target_pose))
        return {"status": "Success"}

    def execute_actions(
        self,
        env_id,
        action_type,
        actions,
        mutation_id,
        expected_version=None,
    ):
        self.calls.append(
            (
                "execute_actions",
                env_id,
                action_type,
                mutation_id,
                expected_version,
            )
        )
        return {"state": "completed"}

    def execute_qpos_updates(
        self,
        env_id,
        updates,
        mutation_id,
        expected_plan=None,
    ):
        self.calls.append(
            (
                "execute_qpos_updates",
                env_id,
                updates,
                mutation_id,
                expected_plan,
            )
        )
        return {"state": "completed"}

    def reset_episode(self, env_id, seed, mutation_id, reset_options=None):
        self.calls.append(
            ("reset_episode", env_id, seed, mutation_id, reset_options)
        )
        return {"state": "completed"}

    def get_mutation_result(self, env_id, mutation_id):
        self.calls.append(("get_mutation_result", env_id, mutation_id))
        return {"state": "completed"}


def make_env(*, profile="downloads_hybrid", allow_debug=False):
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.profile = profile
    env.allow_hybrid_debug = allow_debug
    env.venv = FakeVectorEnv()
    return env


def test_standard_profile_remains_default_and_rejects_hybrid_capabilities():
    cfg = SimpleNamespace(auto_reset=False)
    RoboTwinEnv._validate_profile_config("standard", 4, cfg.auto_reset)

    env = make_env(profile="standard")
    with pytest.raises(RuntimeError, match="downloads_hybrid"):
        env.get_robot_state()


@pytest.mark.parametrize(
    ("num_envs", "auto_reset", "message"),
    [
        (2, False, "num_envs"),
        (1, True, "auto_reset"),
    ],
)
def test_hybrid_profile_validates_single_env_without_auto_reset(
    num_envs, auto_reset, message
):
    with pytest.raises(ValueError, match=message):
        RoboTwinEnv._validate_profile_config("downloads_hybrid", num_envs, auto_reset)


def test_capability_handshake_contains_full_contract():
    env = make_env()
    capabilities = env.get_capabilities()

    assert capabilities["contract_version"] == "robotwin-agent-v1"
    assert capabilities["profile"] == "downloads_hybrid"
    assert capabilities["action_specs"]["qpos"]["shape"] == [14]
    assert capabilities["action_specs"]["ee"]["shape"] == [16]
    assert capabilities["action_specs"]["ee"]["quaternion_order"] == "wxyz"
    assert capabilities["model_contract"]["default_use_length"] == 50
    assert capabilities["episode_budget_counter"] == "take_action_cnt"


def test_capability_handshake_rejects_wrong_native_snapshot():
    env = make_env()
    env.venv = FakeVectorEnv(compatibility_id="wrong")

    with pytest.raises(RuntimeError, match="compatibility mismatch"):
        env.get_capabilities()


def test_capability_handshake_rejects_wrong_asset_snapshot():
    env = make_env()
    native = env.venv.get_capabilities()
    native["asset_snapshot"]["tree_sha256"] = "wrong"
    env.venv.get_capabilities = lambda: native

    with pytest.raises(RuntimeError, match="asset snapshot contract mismatch"):
        env.get_capabilities()


def test_capability_handshake_rejects_incomplete_native_surface():
    env = make_env()
    env.venv.get_capabilities = lambda: {
        "compatibility_id": ROBOTWIN_COMPATIBILITY_ID,
        "asset_snapshot": {
            "file_count": 20854,
            "tree_sha256": (
                "6ef76bdd0b4b8fefbc5d8dc855563e3b4c4c03674c586079141ab2b66079c12b"
            ),
        },
        "server_instance_id": "server-1",
        "mutation_id_prefix": "server-1:",
        "capabilities": ["robot_state"],
    }

    with pytest.raises(RuntimeError, match="missing capabilities"):
        env.get_capabilities()


def test_debug_state_requires_explicit_configuration():
    env = make_env(allow_debug=False)
    with pytest.raises(PermissionError, match="disabled"):
        env.capture_debug_state()

    env.allow_hybrid_debug = True
    assert env.capture_debug_state() == {"actors": {}}


def test_mutation_forwarding_preserves_id_and_expected_version():
    env = make_env()
    version = {"episode_generation": 3, "mutation_seq": 9}
    result = env.execute_actions(
        0,
        "ee",
        np.zeros((2, 16)),
        "episode-3-lingbot-1",
        version,
    )

    assert result == {"state": "completed"}
    assert env.venv.calls[-1] == (
        "execute_actions",
        0,
        "ee",
        "episode-3-lingbot-1",
        version,
    )


def test_public_hybrid_capabilities_forward_only_through_vector_env():
    env = make_env()
    target_pose = np.zeros(7)
    updates = [{"arm": "left", "gripper": 0.5}]
    version = {"episode_generation": 3, "mutation_seq": 9}

    assert env.capture_policy_observation(0) == {"task": "place the fan"}
    assert env.capture_agent_observation(0) == {"frame_id": 1}
    assert env.get_episode_status(0) == {"success": False}
    assert env.plan_arm_path(0, "left", target_pose) == {"status": "Success"}
    assert env.execute_qpos_updates(
        0,
        updates,
        "server-1:qpos",
        version,
    ) == {"state": "completed"}
    assert env.reset_episode(
        0,
        100002,
        "server-1:reset",
        {"exact_seed": True},
    ) == {"state": "completed"}
    assert env.get_mutation_result(0, "server-1:qpos") == {
        "state": "completed"
    }

    assert env.venv.calls == [
        ("capture_policy_observation", 0),
        ("capture_agent_observation", 0),
        ("get_episode_status", 0),
        ("plan_arm_path", 0, "left", target_pose),
        ("execute_qpos_updates", 0, updates, "server-1:qpos", version),
        (
            "reset_episode",
            0,
            100002,
            "server-1:reset",
            {"exact_seed": True},
        ),
        ("get_mutation_result", 0, "server-1:qpos"),
    ]


def test_hybrid_profile_is_forwarded_to_native_vector_env(monkeypatch):
    constructed = {}

    class VectorEnv:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    robotwin_module = ModuleType("robotwin")
    robotwin_envs_module = ModuleType("robotwin.envs")
    vector_env_module = ModuleType("robotwin.envs.vector_env")
    vector_env_module.VectorEnv = VectorEnv
    monkeypatch.setitem(sys.modules, "robotwin", robotwin_module)
    monkeypatch.setitem(sys.modules, "robotwin.envs", robotwin_envs_module)
    monkeypatch.setitem(sys.modules, "robotwin.envs.vector_env", vector_env_module)
    monkeypatch.setattr(
        robotwin_env_module.mp, "set_start_method", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        robotwin_env_module.OmegaConf,
        "to_container",
        lambda value, resolve: value,
    )

    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.profile = "downloads_hybrid"
    env.seed = 100002
    env.num_envs = 1
    env.cfg = SimpleNamespace(
        assets_path="/robotwin",
        task_config={"task_name": "place_fan"},
        get=lambda key, default=None: (
            100002 if key == "hybrid_initial_seed" else default
        ),
    )
    env._init_env()

    assert constructed["profile"] == "downloads_hybrid"
    assert constructed["n_envs"] == 1
    assert constructed["env_seeds"] == [100002]


def test_standard_profile_keeps_legacy_vector_env_constructor(monkeypatch):
    constructed = {}

    class VectorEnv:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    robotwin_module = ModuleType("robotwin")
    robotwin_envs_module = ModuleType("robotwin.envs")
    vector_env_module = ModuleType("robotwin.envs.vector_env")
    vector_env_module.VectorEnv = VectorEnv
    monkeypatch.setitem(sys.modules, "robotwin", robotwin_module)
    monkeypatch.setitem(sys.modules, "robotwin.envs", robotwin_envs_module)
    monkeypatch.setitem(sys.modules, "robotwin.envs.vector_env", vector_env_module)
    monkeypatch.setattr(
        robotwin_env_module.mp, "set_start_method", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        robotwin_env_module.OmegaConf,
        "to_container",
        lambda value, resolve: value,
    )

    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.profile = "standard"
    env.seed = 7
    env.num_envs = 2
    env.reset_state_ids = np.asarray([11, 12])
    env.cfg = SimpleNamespace(
        assets_path="/robotwin",
        task_config={"task_name": "place_empty_cup"},
    )
    env._init_env()

    assert constructed == {
        "task_config": {"task_name": "place_empty_cup"},
        "n_envs": 2,
        "env_seeds": [11, 12],
    }
