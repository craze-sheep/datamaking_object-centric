#!/usr/bin/env bash
set -Eeuo pipefail

# 清除代理，直连 hf-mirror
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy no_proxy NO_PROXY
export HF_ENDPOINT=https://hf-mirror.com

TAR_DIR="/home/lzy/project/slot-datamaking/tars"
STATE_DIR="/home/lzy/project/slot-datamaking/.hf_upload_state"
OWNER="craze-sheep"

mkdir -p "$STATE_DIR"

for s in S1 S2 S3 S4 S5 S6 S7 S8; do
  marker="$STATE_DIR/${s}.uploaded"
  tar_file="$TAR_DIR/${s}.tar.gz"

  if [[ -f "$marker" ]]; then
    echo "[skip] $s already uploaded"
    continue
  fi

  if [[ ! -s "$tar_file" ]]; then
    echo "[error] $s tar missing or empty: $tar_file"
    continue
  fi

  repo="${OWNER}/slot-datamaking-${s,,}"
  size=$(du -h "$tar_file" | cut -f1)
  echo ""
  echo "========================================"
  echo "[$(date '+%F %T')] Uploading $s ($size) -> $repo"
  echo "========================================"

  for attempt in 1 2 3; do
    if hf upload "$repo" "$tar_file" "/${s}.tar.gz" --type dataset 2>&1; then
      touch "$marker"
      echo "[$(date '+%F %T')] ✓ $s done"
      break
    else
      echo "[attempt $attempt/3] $s failed, retrying in 30s..."
      sleep 30
    fi
  done
done

echo ""
echo "=== Final status ==="
for s in S1 S2 S3 S4 S5 S6 S7 S8; do
  if [[ -f "$STATE_DIR/${s}.uploaded" ]]; then
    echo "  $s ✅"
  else
    echo "  $s ❌"
  fi
done
