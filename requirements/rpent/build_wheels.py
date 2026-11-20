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

"""Build traceable CUDA 12 wheel variants without changing native kernels."""

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
ROOT = parser.parse_args().package.resolve()


def rebuild(name, version, transform):
    original = next((ROOT / "sources").glob(name + "-*.whl"))
    with zipfile.ZipFile(original) as archive:
        entries = {
            n: archive.read(n) for n in archive.namelist() if not n.endswith("/")
        }
    metadata = next(n for n in entries if n.endswith(".dist-info/METADATA"))
    old_info = metadata.rsplit("/", 1)[0]
    new_info = name + "-" + version + ".dist-info"
    entries = {
        n.replace(old_info, new_info): v
        for n, v in entries.items()
        if not n.endswith(("/RECORD", "/RECORD.jws", "/RECORD.p7s"))
    }
    meta_name = new_info + "/METADATA"
    meta = re.sub(
        r"(?m)^Version: .+$", "Version: " + version, entries[meta_name].decode()
    )
    entries, meta = transform(entries, meta)
    entries[meta_name] = meta.encode()
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    entries[new_info + "/rpent-build.json"] = json.dumps(
        {
            "base_wheel": original.name,
            "base_sha256": original_hash,
            "version": version,
        },
        indent=2,
    ).encode()
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for path, data in sorted(entries.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        )
        writer.writerow([path, "sha256=" + digest, len(data)])
    writer.writerow([new_info + "/RECORD", "", ""])
    entries[new_info + "/RECORD"] = record.getvalue().encode()
    suffix = original.name.split("-", 2)[2]
    out = ROOT / "wheelhouse" / f"{name}-{version}-{suffix}"
    temporary = out.with_suffix(".building")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(entries.items()):
            archive.writestr(path, data)
    temporary.replace(out)
    print("BUILT", out.name, out.stat().st_size, flush=True)


def bridge(entries, metadata):
    metadata = metadata.replace("megatron-core[dev,mlm]", "megatron-core[mlm]")
    metadata = metadata.replace(
        "Requires-Dist: flash-linear-attention\n", "Requires-Dist: fla-core==0.6.0\n"
    )
    metadata = metadata.replace(
        "hydra-core<=1.3.2,>1.3", "hydra-core>=1.3.2,<1.4.0.dev8"
    )
    metadata = metadata.replace(
        "flashinfer-python==0.6.8.post1", "flashinfer-python==0.6.11.post1"
    )
    metadata = metadata.replace(
        "flashinfer-cubin==0.6.8.post1", "flashinfer-cubin==0.6.11.post1"
    )
    return entries, metadata


def mcore(entries, metadata):
    directory = ROOT / "build/mcore-cpu-offload"
    for name, data in entries.items():
        if name.startswith("megatron/"):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    patch = ROOT / "sources/rpent-adapters/megatron-core-r0.18.0-cpu-offload.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=directory, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=directory, check=True)
    for name in [
        "megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py",
        "megatron/core/optimizer/distrib_optimizer.py",
    ]:
        entries[name] = (directory / name).read_bytes()
    name = "megatron/core/dist_checkpointing/strategies/nvrx.py"
    text = (
        entries[name]
        .decode()
        .replace(
            "from importlib import import_module",
            "from importlib import import_module\nfrom importlib.metadata import version as distribution_version",
        )
    )
    text = text.replace(
        "    try:\n        core = import_module(",
        "    if not is_nvrx_min_version():\n        return False\n\n    try:\n        core = import_module(",
    )
    text = text.replace(
        'str(nvrx.__version__) if HAVE_NVRX else "0.0.0"',
        'distribution_version("nvidia-resiliency-ext") if HAVE_NVRX else "0.0.0"',
    )
    entries[name] = text.encode()
    return entries, metadata


shutil.copy2(next((ROOT / "sources").glob("sglang-*.whl")), ROOT / "wheelhouse")

rebuild("megatron_bridge", "0.5.0+rpent.cu128.1", bridge)
rebuild("megatron_core", "0.18.0+rpent.cu128.1", mcore)
for name, release in (
    ("rlinf", "0.4.0"),
    ("rpent", "0.1.0"),
    ("sam3", "0.1.0"),
    ("rpent_libero", "0.2.0"),
    ("rpent_liberopro", "0.2.0"),
):
    rebuild(
        name, release + "+rpent.h200.1", lambda entries, metadata: (entries, metadata)
    )

base = json.loads((ROOT / "manifests/base-h200_1.json").read_text())
overrides = {
    "sglang": "0.5.12.post1",
    "megatron-bridge": "0.5.0+rpent.cu128.1",
    "numpy": "1.26.4",
    "transformers": "5.8.1",
    "peft": "0.18.1",
    "timm": "1.0.27",
    "mcp": "1.26.0",
    "grpcio": "1.81.1",
    "grpcio-health-checking": "1.81.1",
    "grpcio-reflection": "1.81.1",
    "setuptools": "80.9.0",
    "scipy": "1.15.3",
    "ml-dtypes": "0.5.4",
    "anthropic": "1.9.0",
    "openai": "3.19.0",
    "markdown-it-py": "3.0.0",
}
overrides.update(
    {
        "rlinf": "0.4.0+rpent.h200.1",
        "rpent": "0.1.0+rpent.h200.1",
        "sam3": "0.1.0+rpent.h200.1",
        "rpent-libero": "0.2.0+rpent.h200.1",
        "rpent-liberopro": "0.2.0+rpent.h200.1",
    }
)
requirements = []
for row in base["packages"]:
    name = row["name"].lower().replace("_", "-")
    # These prebuilt image extensions are retained and checked by the installer.
    if name in {"apex", "flash-attn", "sglang-kernel", "sgl-deep-gemm"}:
        continue
    requirements.append(f"{name}=={overrides.get(name, row['version'])}")
requirements += [
    "megatron-core==0.18.0+rpent.cu128.1",
    "fla-core==0.6.0",
    "mujoco==3.3.0",
    "decord==0.6.0",
    "robosuite==1.5.2",
    "opencv-python==4.11.0.86",
    "gym==0.26.2",
    "gymnasium==1.4.0",
    "bddl",
    "h5py",
    "matplotlib",
    "imageio[ffmpeg]",
    "pycocotools",
    "ftfy==6.1.1",
    "iopath",
    "termcolor",
    "pydantic-ai-slim[anthropic,openai,cli]==2.53.0",
    "pydantic-ai-harness==0.53.0",
    "claude-agent-sdk>=0.1.60",
    "openai-codex>=0.1.0b3",
    "pytest-timeout",
    "coverage",
    "rlinf==0.4.0+rpent.h200.1",
    "rpent==0.1.0+rpent.h200.1",
    "sam3==0.1.0+rpent.h200.1",
    "rpent-libero==0.2.0+rpent.h200.1",
    "rpent-liberopro==0.2.0+rpent.h200.1",
    "Sphinx==8.1.3",
    "pydata-sphinx-theme==0.16.1",
    "sphinx-design==0.7.0",
    "sphinx-copybutton==0.5.2",
    "myst-parser==4.0.1",
    "recommonmark==0.7.1",
    "sphinx-last-updated-by-git==0.3.8",
    "sphinx-markdown-tables==0.0.17",
    "sphinx-rtd-theme==3.0.2",
    "sphinx-sitemap==2.7.2",
    "sphinxcontrib-video==0.4.1",
    "sphinx-reredirects==1.1.0",
]
(ROOT / "locks/requirements.in").write_text("\n".join(sorted(requirements)) + "\n")
print("REQUIREMENTS_READY", len(requirements), flush=True)
