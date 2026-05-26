#!/usr/bin/env bash
set -Eeuo pipefail

DB_DIR="/home/lzy/project/slot-datamaking/database1"
TAR_DIR="/home/lzy/project/slot-datamaking/tars"
STATE_DIR="/home/lzy/project/slot-datamaking/.hf_upload_state"
OWNER="craze-sheep"
SLOTS=(S1 S2 S3 S4 S5 S6 S7 S8)

mkdir -p "$TAR_DIR" "$STATE_DIR"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

create_tar_if_needed() {
  local slot="$1"
  local tar_file="$TAR_DIR/${slot}.tar.gz"
  local tmp_file="${tar_file}.tmp.$$"

  if [[ -s "$tar_file" ]]; then
    log "$slot tar exists, skip: $tar_file"
    return 0
  fi

  if [[ ! -d "$DB_DIR/$slot" ]]; then
    log "$slot source directory missing: $DB_DIR/$slot"
    return 1
  fi

  rm -f "$tmp_file"
  log "$slot packing -> $tar_file"

  if command -v pigz >/dev/null 2>&1; then
    tar -C "$DB_DIR" -cf - "$slot" | pigz -1 > "$tmp_file"
  else
    tar -C "$DB_DIR" -czf "$tmp_file" "$slot"
  fi

  mv "$tmp_file" "$tar_file"
  log "$slot packed"
}

pack_all() {
  for slot in "${SLOTS[@]}"; do
    create_tar_if_needed "$slot" || return 1
  done
}

upload_slot() {
  local slot="$1"
  local tar_file="$TAR_DIR/${slot}.tar.gz"
  local marker="$STATE_DIR/${slot}.uploaded"
  local repo="${OWNER}/slot-datamaking-${slot,,}"

  if [[ -f "$marker" ]]; then
    return 0
  fi

  log "$slot uploading -> $repo"
  hf upload "$repo" "$tar_file" "/${slot}.tar.gz" --type dataset
  touch "$marker"
  log "$slot uploaded"
}

all_uploaded() {
  local slot
  for slot in "${SLOTS[@]}"; do
    [[ -f "$STATE_DIR/${slot}.uploaded" ]] || return 1
  done
  return 0
}

upload_when_ready() {
  local pack_status_file="$1"
  local slot tar_file marker made_progress pack_status

  while ! all_uploaded; do
    made_progress=0

    for slot in "${SLOTS[@]}"; do
      tar_file="$TAR_DIR/${slot}.tar.gz"
      marker="$STATE_DIR/${slot}.uploaded"

      if [[ ! -f "$marker" && -s "$tar_file" ]]; then
        upload_slot "$slot"
        made_progress=1
      fi
    done

    if all_uploaded; then
      break
    fi

    if [[ -f "$pack_status_file" && "$made_progress" -eq 0 ]]; then
      pack_status="$(<"$pack_status_file")"
      if [[ "$pack_status" != "0" ]]; then
        log "packaging failed"
        return "$pack_status"
      fi

      log "packaging finished, but some uploads are still missing tar files"
      return 1
    fi

    sleep 10
  done
}

main() {
  command -v hf >/dev/null 2>&1 || {
    log "missing command: hf"
    exit 1
  }

  local pack_status_file="$STATE_DIR/.pack_status.$$"

  rm -f "$pack_status_file"
  (
    trap 'status=$?; printf "%s\n" "$status" > "$pack_status_file"' EXIT
    pack_all
  ) &
  local pack_pid=$!

  trap 'kill "$pack_pid" 2>/dev/null || true; rm -f "$pack_status_file"' EXIT INT TERM

  upload_when_ready "$pack_status_file"
  wait "$pack_pid"
  rm -f "$pack_status_file"

  log "done"
}

main "$@"
