# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Build the narrow FLA 0.6.0 TileLang singleton-stride compatibility wheel."""

import argparse
import base64
import csv
import difflib
import hashlib
import io
import json
import zipfile
from pathlib import Path

ORIGINAL_SHA256 = "b0708b70e0b119cd58a0a5146ea56cb175cef6ed969d8ac60267e3e077ce3ebd"
VERSION = "0.6.0+rpent.layout1"


def main() -> None:
    """Verify the official wheel and preserve every unrelated package file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.original.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Expected the recorded official fla-core 0.6.0 wheel")
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.original) as archive:
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    source = "fla/ops/common/backends/tilelang/chunk_bwd.py"
    original = files[source].decode()
    anchor = "    B, T, H, K = k.shape\n    HV, V = v.shape[2], v.shape[-1]\n"
    if original.count(anchor) != 1:
        raise ValueError("Unexpected TileLang backward source")
    patched = original.replace(
        anchor,
        "    # Singleton dimensions can be contiguous with noncanonical strides.\n"
        "    v = v.reshape(-1).view(v.shape)\n" + anchor,
    )
    files[source] = patched.encode()
    old_info, new_info = "fla_core-0.6.0.dist-info", f"fla_core-{VERSION}.dist-info"
    files["fla/__init__.py"] = files["fla/__init__.py"].replace(
        b'__version__ = "0.6.0"', f'__version__ = "{VERSION}"'.encode()
    )
    files[f"{old_info}/METADATA"] = files[f"{old_info}/METADATA"].replace(
        b"Version: 0.6.0\n", f"Version: {VERSION}\n".encode()
    )
    files = {
        name.replace(old_info, new_info): data
        for name, data in files.items()
        if name != f"{old_info}/RECORD"
    }
    rows = [
        [
            name,
            "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .decode()
            .rstrip("="),
            len(data),
        ]
        for name, data in sorted(files.items())
    ]
    rows.append([f"{new_info}/RECORD", "", ""])
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[f"{new_info}/RECORD"] = record.getvalue().encode()
    wheel = args.output / f"fla_core-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "x") as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 11, 21, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    (args.output / "layout.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                patched.splitlines(True),
                fromfile="a/" + source,
                tofile="b/" + source,
            )
        )
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "original_sha256": ORIGINAL_SHA256,
                "wheel": wheel.name,
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "original_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "patched_source_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                "core_bridge_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
