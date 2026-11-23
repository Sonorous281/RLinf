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

"""Resolve native cuRobo asset templates without importing a simulator or Ray."""

import os
from pathlib import Path

import yaml


def curobo_config_resolves(config: Path) -> bool:
    """Whether a curobo config's asset root and URDF exist on this machine."""
    try:
        kinematics = yaml.safe_load(config.read_text())["robot_cfg"]["kinematics"]
        asset_root = kinematics["asset_root_path"]
        urdf_path = kinematics["urdf_path"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return False
    return os.path.isdir(asset_root) and os.path.isfile(urdf_path)


def create_curobo_configs(assets_path: str) -> None:
    """Make each robot's ``curobo.yml`` readable by the runtime.

    The dataset publishes ``curobo_tmp.yml`` templates whose ``${ASSETS_PATH}``
    placeholder the runtime does not expand, while the runtime opens
    ``curobo.yml`` directly. Render a missing file, and rerender one whose paths
    do not resolve (for example after moving the asset tree); keep a usable file
    because it may carry local edits.
    """
    for robot in sorted((Path(assets_path) / "Assets" / "Robots").iterdir()):
        if not robot.is_dir():
            continue
        config = robot / "curobo.yml"
        if config.is_file() and curobo_config_resolves(config):
            continue
        template = robot / "curobo_tmp.yml"
        if not template.is_file():
            raise FileNotFoundError(
                f"{robot} has no usable curobo.yml and no curobo_tmp.yml "
                "template; the RoboDojo runtime cannot build its motion planner."
            )
        content = template.read_text().replace("${ASSETS_PATH}", assets_path)
        temporary = config.with_name(f".{config.name}.{os.getpid()}.tmp")
        temporary.write_text(content)
        if config.exists():
            os.replace(temporary, config)
        else:
            try:
                os.link(temporary, config)
            except FileExistsError:
                pass
            finally:
                temporary.unlink()
