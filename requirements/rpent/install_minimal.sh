#!/usr/bin/env bash
set -eo pipefail
source /opt/venv/reason/bin/activate
set -u
PACKAGE_ROOT="${1:?Usage: install_minimal.sh /path/to/offline/release}"
shift
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /opt/venv/reason/bin/python "$SCRIPT_ROOT/install_minimal.py" --package "$PACKAGE_ROOT" "$@"
