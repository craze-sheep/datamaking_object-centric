#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: docker run ... kubric-gpu <script.py> [args...]"
    exit 1
fi

SCRIPT="$1"
shift

# Blender 自带的库（libIex, libOpenEXR 等）
export LD_LIBRARY_PATH="/opt/blender-3.6.15-linux-x64/lib:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${PYTHONPATH:-}"
if [ -d "/kubric" ]; then
    export PYTHONPATH="/kubric:${PYTHONPATH}"
fi

export KUBRIC_USE_GPU="${KUBRIC_USE_GPU:-True}"

exec blender --background --python "$SCRIPT" -- "$@"
