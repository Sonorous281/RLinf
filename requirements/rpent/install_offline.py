# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install a checked wheelhouse into the existing reason image environment."""

import argparse
import ctypes
import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

PROTECTED = (
    "torch",
    "torchvision",
    "flash-attn",
    "transformer-engine",
    "transformer-engine-torch",
    "transformer-engine-cu12",
    "triton",
    "apex",
    "sglang-kernel",
    "sgl-deep-gemm",
)


def installed() -> dict[str, str]:
    """Return the environment's distribution versions under normalized names."""
    return {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in metadata.distributions()
    }


def backup_packages(destination: Path, selected: list[str]) -> None:
    """Keep original package files, including their entry points, before replacement."""
    with zipfile.ZipFile(
        destination, "w", zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        seen = set()
        prefix = Path(sys.prefix).resolve()
        for name in selected:
            try:
                distribution = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                continue
            for relative in distribution.files or []:
                path = Path(distribution.locate_file(relative)).resolve()
                if not path.is_file() or path in seen or path.suffix == ".pyc":
                    continue
                seen.add(path)
                if not path.is_relative_to(prefix):
                    raise RuntimeError(
                        f"Package file outside the target environment: {path}"
                    )
                archive.write(path, str(path.relative_to(prefix)))


def wheel_metadata(path: Path) -> str:
    """Read the exact metadata of the wheel selected for this ABI."""
    with zipfile.ZipFile(path) as archive:
        member = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        return archive.read(member).decode()


def wheel_sources_match(name: str, wheel: Path) -> bool:
    """Compare local-build Python files, including same-version development rebuilds."""
    distribution = metadata.distribution(name)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.endswith(".py") or ".dist-info/" in member:
                continue
            installed_file = Path(distribution.locate_file(member))
            if (
                not installed_file.is_file()
                or installed_file.read_bytes() != archive.read(member)
            ):
                return False
    return True


def link_assets(root: Path) -> None:
    """Connect installed benchmark packages to the shared, separately stored assets."""
    stack = root.parent.parent
    for distribution, module, data in (
        ("rpent-libero", "libero/libero", "libero"),
        ("rpent-liberopro", "liberopro/liberopro", "libero-pro"),
    ):
        target = stack / "data" / data / "assets"
        if not target.is_dir():
            raise RuntimeError(f"Shared benchmark assets not found: {target}")
        package = Path(metadata.distribution(distribution).locate_file(module))
        link_tree(target, package / "assets")


def link_tree(source: Path, destination: Path) -> None:
    """Keep installed asset modules and link missing scene data from shared storage."""
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing asset link differs: {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == "__pycache__" or path.suffix in {".py", ".pyc"}:
            continue
        target = destination / path.name
        if path.is_dir() and target.is_dir() and not target.is_symlink():
            link_tree(path, target)
        elif not target.exists() and not target.is_symlink():
            target.symlink_to(path, target_is_directory=path.is_dir())


def main() -> None:
    """Validate the image, install only local wheels, then check the final environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.package.resolve()
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError(
            "This package requires the Linux x86_64 Python 3.12 reason image"
        )
    if Path(sys.prefix) != Path("/opt/venv/reason"):
        raise RuntimeError(f"Expected /opt/venv/reason, found {sys.prefix}")
    for library in ("libEGL.so.1", "libGL.so.1", "libOSMesa.so.8"):
        ctypes.CDLL(library)
    baseline = json.loads((root / "manifests/base-h200_1.json").read_text())
    base = {
        x["name"].lower().replace("_", "-"): x["version"] for x in baseline["packages"]
    }
    before = installed()
    for name in PROTECTED:
        if before.get(name) != base[name]:
            raise RuntimeError(
                f"Image mismatch for {name}: expected {base[name]}, got {before.get(name)}"
            )
    if "flash-attn-4" in before:
        raise RuntimeError("Remove FA4 before using this FA2 image package")
    manifest = json.loads((root / "manifests/package-files.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise RuntimeError(f"Invalid package path: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Offline package checksum mismatch: {name}")
    for benchmark in ("libero", "libero-pro"):
        assets = root.parent.parent / "data" / benchmark / "assets"
        if not assets.is_dir():
            raise RuntimeError(f"Shared benchmark assets not found: {assets}")
    if args.check_only:
        print("Offline package and protected image packages verified", flush=True)
        return
    wanted = json.loads((root / "locks/runtime-versions.json").read_text())
    wheels = json.loads((root / "manifests/wheels.json").read_text())
    changed = []
    for name, version in wanted.items():
        wheel = root / "wheelhouse" / wheels[name]
        if (
            before.get(name) != version
            or metadata.distribution(name).read_text("METADATA").splitlines()
            != wheel_metadata(wheel).splitlines()
            or ("+rpent." in version and not wheel_sources_match(name, wheel))
        ):
            changed.append(name)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "logs" / f"install-{socket.gethostname()}-{stamp}"
    output.mkdir(parents=True)
    (output / "before.json").write_text(json.dumps(before, indent=2) + "\n")
    if changed:
        backup_packages(output / "replaced-packages.zip", changed)
        environment = {
            **os.environ,
            "PIP_NO_INDEX": "1",
            "UV_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        for name in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "all_proxy",
        ):
            environment.pop(name, None)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                "--only-binary=:all:",
                "--find-links",
                str(root / "wheelhouse"),
                *[str(root / "wheelhouse" / wheels[name]) for name in changed],
            ],
            check=True,
            env=environment,
        )
    link_assets(root)
    after = installed()
    for name, expected in wanted.items():
        if after.get(name) != expected:
            raise RuntimeError(
                f"Installed version mismatch: {name}={after.get(name)}, expected {expected}"
            )
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    (output / "pip-check.txt").write_text(pip_check.stdout + pip_check.stderr)
    (output / "after.json").write_text(json.dumps(after, indent=2) + "\n")
    receipt = {
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "prefix": sys.prefix,
        "changed_packages": changed,
        "package_manifest_sha256": hashlib.sha256(
            (root / "manifests/package-files.json").read_bytes()
        ).hexdigest(),
        "versions_sha256": hashlib.sha256(
            json.dumps(after, sort_keys=True).encode()
        ).hexdigest(),
        "pip_check_returncode": pip_check.returncode,
        "passed": True,
    }
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
