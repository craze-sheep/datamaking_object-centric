#!/usr/bin/env bash
set -Eeuo pipefail

# 走代理直连 huggingface.co（之前验证过 git clone 走代理可以通）
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897

TAR_DIR="/home/lzy/project/slot-datamaking/tars"
STATE_DIR="/home/lzy/project/slot-datamaking/.hf_upload_state"
WORK_DIR="/home/lzy/project/slot-datamaking/.hf_upload_work"
OWNER="craze-sheep"

mkdir -p "$STATE_DIR" "$WORK_DIR"

for s in S1 S2 S3 S4 S5 S6 S7 S8; do
  marker="$STATE_DIR/${s}.uploaded"
  tar_file="$TAR_DIR/${s}.tar.gz"

  if [[ -f "$marker" ]]; then
    echo "[skip] $s already uploaded"
    continue
  fi

  if [[ ! -s "$tar_file" ]]; then
    echo "[error] $s tar missing: $tar_file"
    continue
  fi

  repo="${OWNER}/slot-datamaking-${s,,}"
  repo_dir="$WORK_DIR/slot-datamaking-${s,,}"
  size=$(du -h "$tar_file" | cut -f1)

  echo ""
  echo "========================================"
  echo "[$(date '+%F %T')] Uploading $s ($size) -> $repo"
  echo "========================================"

  # 清理旧的 clone
  rm -rf "$repo_dir"

  # clone repo
  echo "Cloning $repo..."
  if ! git clone "https://huggingface.co/datasets/${repo}" "$repo_dir" 2>&1; then
    echo "[error] clone failed for $s"
    continue
  fi

  # 启用 lfs
  cd "$repo_dir"
  git lfs install

  # 复制 tar 文件
  echo "Copying $tar_file..."
  cp "$tar_file" "./${s}.tar.gz"

  # track tar 文件
  git lfs track "*.tar.gz"
  git add .gitattributes "${s}.tar.gz"

  # commit
  git commit -m "Add ${s}.tar.gz dataset" 2>&1

  # push（带重试）
  echo "Pushing..."
  pushed=0
  for attempt in 1 2 3; do
    if git push 2>&1; then
      pushed=1
      break
    else
      echo "[attempt $attempt/3] push failed, retrying in 30s..."
      sleep 30
    fi
  done

  if [[ $pushed -eq 1 ]]; then
    touch "$marker"
    echo "[$(date '+%F %T')] ✓ $s done"
  else
    echo "[error] $s push failed after 3 attempts"
  fi

  # 清理 clone 目录省空间
  cd /
  rm -rf "$repo_dir"
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
