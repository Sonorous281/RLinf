#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /opt/venv/reason/bin/python "$PACKAGE_ROOT/tools/install_offline.py" --package "$PACKAGE_ROOT" "$@"
