#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

DIR="$ROOT_DIR/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16"
URL_BASE="https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-bf16/resolve/main"

download_shard() {
  local shard="$1"
  echo "[$(date +%H:%M:%S)] Resuming $shard..."
  until curl -L -C - --fail \
    --retry 50 --retry-delay 3 --retry-all-errors \
    --connect-timeout 20 --speed-time 30 --speed-limit 10240 \
    -H "Authorization: Bearer ${HF_TOKEN:-}" \
    -o "$DIR/$shard.part" \
    "$URL_BASE/$shard"; do
    echo "[$(date +%H:%M:%S)] Connection reset on $shard — resuming from $(ls -lh "$DIR/$shard.part" 2>/dev/null | awk '{print $5}') in 3s..."
    sleep 3
  done
  mv "$DIR/$shard.part" "$DIR/$shard"
  echo "[$(date +%H:%M:%S)] Completed $shard!"
}

for s in \
  model-00009-of-00013.safetensors \
  model-00010-of-00013.safetensors \
  model-00011-of-00013.safetensors \
  model-00012-of-00013.safetensors \
  model-00013-of-00013.safetensors; do
  if [ ! -f "$DIR/$s" ]; then
    download_shard "$s"
  else
    echo "[$(date +%H:%M:%S)] $s already exists, skipping."
  fi
done

echo "All 13 shards downloaded in $DIR"
