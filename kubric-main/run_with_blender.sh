#!/usr/bin/env bash
set -euo pipefail

KUBRIC_ENV="${KUBRIC_ENV:-/root/miniconda3/envs/kubric39}"
KUBRIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="${KUBRIC_ENV}/lib/python3.9/site-packages"
BLENDER_BIN="${BLENDER_BIN:-/root/project/blender-3.0.1-linux-x64/blender}"

export PYTHONPATH="${KUBRIC_ROOT}:${SITE_PACKAGES}:${PYTHONPATH:-}"
exec "${BLENDER_BIN}" --background --python "$@"
