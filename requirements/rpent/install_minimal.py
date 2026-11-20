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

"""Install an explicit offline wheel list while preserving the image's Core/Bridge."""

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path


def link_assets(source: Path, target: Path) -> None:
    """Link shared benchmark data without overriding installed Python modules."""
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Asset link {target} does not point to {source}")
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "__pycache__" or item.suffix in {".py", ".pyc"}:
            continue
        dest = target / item.name
        if item.is_dir() and dest.is_dir() and not dest.is_symlink():
            link_assets(item, dest)
        elif not dest.exists() and not dest.is_symlink():
            dest.symlink_to(item, target_is_directory=item.is_dir())


def snapshot(protected: list[str]) -> dict:
    """Record package versions, protected payload hashes and original Core state."""
    result = {
        "packages": {
            d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in metadata.distributions()
        },
        "protected": {},
    }
    for name in protected:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            result["protected"][name] = None
            continue
        digest = hashlib.sha256()
        for item in sorted(dist.files or [], key=str):
            path = dist.locate_file(item)
            if path.is_file() and not str(item).endswith((".pyc", ".pyo")):
                digest.update(str(item).encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        result["protected"][name] = {
            "version": dist.version,
            "digest": digest.hexdigest(),
        }
    core = "/opt/venv/reason/Megatron-LM"
    result["core_head"] = subprocess.check_output(
        ["git", "-C", core, "rev-parse", "HEAD"], text=True
    ).strip()
    result["core_status"] = subprocess.check_output(
        ["git", "-C", core, "status", "--porcelain"], text=True
    )
    return result


def installed_payload_matches(row: dict, path: Path) -> bool:
    """Compare the wheel's Python payload with its installed distribution."""
    try:
        dist = metadata.distribution(row["name"])
    except metadata.PackageNotFoundError:
        return False
    if dist.version != row["version"]:
        return False
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".py"):
                installed = dist.locate_file(name)
                if (
                    not installed.is_file()
                    or hashlib.sha256(installed.read_bytes()).digest()
                    != hashlib.sha256(z.read(name)).digest()
                ):
                    return False
    return True


def main() -> None:
    """Validate and install a release without dependency resolution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if sys.prefix != "/opt/venv/reason" or sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Expected /opt/venv/reason Python 3.12, got {sys.prefix}: {sys.version}"
        )
    before = snapshot(manifest["protected"])
    if before["core_head"] != manifest["image_core_commit"] or before["core_status"]:
        raise RuntimeError(
            "Expected the original, clean Core checkout recorded in the manifest"
        )
    for name, expected in manifest.get("image_versions", {}).items():
        actual = before["packages"].get(name)
        if actual != expected:
            raise RuntimeError(
                f"Expected image package {name}=={expected}, found {actual}"
            )
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + socket.gethostname()
    logs = root / "receipts" / stamp
    logs.mkdir(parents=True, exist_ok=False)
    (logs / "before.json").write_text(json.dumps(before, indent=2) + "\n")
    install = []
    for row in manifest["wheels"]:
        if row["name"] in manifest["protected"]:
            raise RuntimeError(f"Cannot install protected image package {row['name']}")
        path = root / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Wheel checksum mismatch: {path}")
        if not installed_payload_matches(row, path):
            if row["name"] in before["packages"] and not row.get("replace_reason"):
                raise RuntimeError(
                    f"Replacement requires a recorded runtime incompatibility: {row['name']}"
                )
            install.append(str(path))
    for resource in manifest.get("resources", []):
        source = root / resource["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != resource["sha256"]:
            raise RuntimeError(f"Resource checksum mismatch: {source}")
        relative = Path(resource["target"])
        if (
            relative.parts[:4] != ("lib", "python3.12", "site-packages", "assets")
            or ".." in relative.parts
        ):
            raise RuntimeError(
                f"Expected a site-packages/assets resource target, got {relative}"
            )
    for entry in manifest.get("asset_links", []):
        if not Path(entry["source"]).is_dir():
            raise RuntimeError(f"Missing shared asset directory: {entry['source']}")
    if install:
        for row in manifest["wheels"]:
            if row["name"] in before["packages"] and str(root / row["file"]) in install:
                dist = metadata.distribution(row["name"])
                backup = logs / "backups" / row["name"]
                for item in dist.files or []:
                    source = dist.locate_file(item)
                    source = source.resolve()
                    if source.is_file() and source.is_relative_to(Path(sys.prefix)):
                        target = backup / source.relative_to(sys.prefix)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                *install,
            ],
            check=True,
        )
    for resource in manifest.get("resources", []):
        source = root / resource["file"]
        relative = Path(resource["target"])
        target = Path(sys.prefix) / relative
        if (
            not target.exists()
            or hashlib.sha256(target.read_bytes()).hexdigest() != resource["sha256"]
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for entry in manifest.get("asset_links", []):
        source = Path(entry["source"])
        link_assets(source, Path(sys.prefix) / entry["target"])
    after = snapshot(manifest["protected"])
    (logs / "after.json").write_text(json.dumps(after, indent=2) + "\n")
    if before["protected"] != after["protected"]:
        raise RuntimeError(
            "A protected image distribution changed; inspect the before/after receipts"
        )
    if before["core_head"] != after["core_head"] or after["core_status"]:
        raise RuntimeError("Original Core changed; inspect the before/after receipts")
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
        check=False,
    )
    (logs / "pip-check.txt").write_text(check.stdout + check.stderr)
    changed = {
        n: {"before": before["packages"].get(n), "after": v}
        for n, v in after["packages"].items()
        if before["packages"].get(n) != v
    }
    receipt = {
        "hostname": socket.gethostname(),
        "time": time.time(),
        "python": sys.executable,
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "changed_packages": changed,
        "installed_wheels": install,
        "protected_unchanged": True,
        "pip_check_returncode": check.returncode,
    }
    (logs / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
