#!/usr/bin/env bash
# Run the medical benchmark (real openly licensed samples) against a real model (issue #1).
#
# Usage:
#   bash scripts/medbench-model-run.sh download   # fetch and verify files
#   bash scripts/medbench-model-run.sh run        # serve the model, run tasks
#   bash scripts/medbench-model-run.sh preflight  # print the chosen load mode
#
# Every download is pinned to a SHA-256, so a changed file fails instead of
# being run. The model is served by llama.cpp's OpenAI-compatible
# llama-server and medbench talks to it over HTTP, so the same script works on
# a Linux CPU runner or an NVIDIA GPU runner. On Apple Silicon use the
# native MLX backend described in docs/apple-silicon.md.
#
# Environment (defaults run MedGemma 4B Q4_K_M on the CPU):
#   CACHE_DIR        where downloads are kept (.cache/medbench)
#   OUT_DIR          where reports and the server log go (medbench-reports)
#   LLAMA_ARCHIVE    llama.cpp release asset name
#   LLAMA_SHA256     its SHA-256
#   CUDART_ARCHIVE   optional CUDA runtime asset (GPU builds only)
#   CUDART_SHA256    its SHA-256
#   MODEL_REPO       Hugging Face repo with GGUF files
#   MODEL_REVISION   commit of that repo
#   MODEL_FILE       language model GGUF and MODEL_SHA256
#   MMPROJ_FILE      vision projector GGUF and MMPROJ_SHA256 (empty: text only)
#   LOAD_MODE        none (copy weights to RAM), mmap, or auto (pick by memory)
#   THREADS          CPU threads (number of CPUs minus two, at least one)
#   GPU_LAYERS       layers to offload to the GPU (0)
#   TASKS            space-separated medbench tasks
#   COUNT            items per task (2)
#   MAX_TOKENS       answer length limit (96)
#   PAGE_MAX_TOKENS  limit for ocr-history, whose pages hold ~500 words (1024)
#   TIME_BUDGET      seconds for all tasks together (900)
#   NICE_LEVEL       CPU priority of the server, 0-19 (10)
#   RUN_PYTEST       1 runs tests/test_real_models.py after the tasks (1)
set -euo pipefail

LLAMA_VERSION="${LLAMA_VERSION:-b11609}"
LLAMA_ARCHIVE="${LLAMA_ARCHIVE:-llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz}"
LLAMA_SHA256="${LLAMA_SHA256:-f95cbb27dd6e377f2ce30f5cfd7ec987af3259c025fc7c98fff62927a16ff528}"
CUDART_ARCHIVE="${CUDART_ARCHIVE:-}"
CUDART_SHA256="${CUDART_SHA256:-}"
MODEL_REPO="${MODEL_REPO:-unsloth/medgemma-4b-it-GGUF}"
MODEL_REVISION="${MODEL_REVISION:-fa17ea2647ef236db91adfe1465a61ab6562cf2c}"
MODEL_FILE="${MODEL_FILE:-medgemma-4b-it-Q4_K_M.gguf}"
MODEL_SHA256="${MODEL_SHA256:-d842e8d2aca3fc5e613c5f9255e693768eeccae729e5c2653159eb79afe751f3}"
MMPROJ_FILE="${MMPROJ_FILE-mmproj-F16.gguf}"
MMPROJ_SHA256="${MMPROJ_SHA256:-1d45f34f8c2f1427a5555f400a63715b3e0c4191341fa2069d5205cb36195c33}"
CACHE_DIR="${CACHE_DIR:-${MEDBENCH_CACHE_DIR:-.cache/medbench}}"
OUT_DIR="${OUT_DIR:-medbench-reports}"
GPU_LAYERS="${GPU_LAYERS:-0}"
TASKS="${TASKS:-diagnosis-ddxplus imaging-xray ocr-prescription}"
COUNT="${COUNT:-2}"
MAX_TOKENS="${MAX_TOKENS:-96}"
PAGE_MAX_TOKENS="${PAGE_MAX_TOKENS:-1024}"
TIME_BUDGET="${TIME_BUDGET:-900}"
NICE_LEVEL="${NICE_LEVEL:-10}"
LOAD_MODE="${LOAD_MODE:-auto}"
PORT="${PORT:-8089}"
THREADS="${THREADS:-$(python3 -c 'import os; print(max(1, (os.cpu_count() or 1) - 2))')}"

LLAMA_DIR="$CACHE_DIR/llama-$LLAMA_VERSION-${LLAMA_SHA256:0:12}"
MODEL_DIR="$CACHE_DIR/models/${MODEL_REVISION:0:12}"

log() {
  printf '[medbench-model-run] %s\n' "$*" >&2
}

# fetch URL DEST SHA256: download once, then always verify the hash.
fetch() {
  local url="$1" dest="$2" sha="$3"
  if [[ ! -f "$dest" ]]; then
    log "downloading $url"
    mkdir -p "$(dirname "$dest")"
    curl --fail --location --retry 3 --silent --show-error \
      --output "$dest.partial" "$url"
    mv "$dest.partial" "$dest"
  fi
  if ! printf '%s  %s\n' "$sha" "$dest" | sha256sum --check --status; then
    log "SHA-256 mismatch for $dest; delete it and update the pinned hash"
    exit 1
  fi
  log "verified $dest"
}

unpack() {
  local archive="$1"
  mkdir -p "$LLAMA_DIR"
  # Release archives keep the binaries in one top-level folder.
  tar --extract --gzip --file "$archive" --directory "$LLAMA_DIR" \
    --strip-components 1 --no-same-owner
}

download() {
  local release="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_VERSION"
  local hub="https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION"
  fetch "$release/$LLAMA_ARCHIVE" "$CACHE_DIR/$LLAMA_ARCHIVE" "$LLAMA_SHA256"
  if [[ ! -x "$LLAMA_DIR/llama-server" ]]; then
    unpack "$CACHE_DIR/$LLAMA_ARCHIVE"
  fi
  if [[ -n "$CUDART_ARCHIVE" ]]; then
    fetch "$release/$CUDART_ARCHIVE" "$CACHE_DIR/$CUDART_ARCHIVE" "$CUDART_SHA256"
    tar --extract --gzip --file "$CACHE_DIR/$CUDART_ARCHIVE" \
      --directory "$LLAMA_DIR" --no-same-owner
  fi
  fetch "$hub/$MODEL_FILE" "$MODEL_DIR/$MODEL_FILE" "$MODEL_SHA256"
  if [[ -n "$MMPROJ_FILE" ]]; then
    fetch "$hub/$MMPROJ_FILE" "$MODEL_DIR/$MMPROJ_FILE" "$MMPROJ_SHA256"
  fi
}

# Memory this process may still use, in bytes: the smaller of MemAvailable
# and the free part of the cgroup limit (containers often cap far below RAM).
available_bytes() {
  if [[ ! -r /proc/meminfo ]]; then
    python3 - "$(dirname "${BASH_SOURCE[0]}")/../src" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from medbench.hardware import detect_host
print(int(detect_host(with_gpus=False).memory_available_gb * 1e9))
PY
    return
  fi
  local available limit current
  available=$(awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
  limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
  if [[ "$limit" =~ ^[0-9]+$ ]]; then
    current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    if ((limit - current < available)); then
      available=$((limit - current))
    fi
  fi
  echo "$available"
}

# With mmap the kernel may drop weight pages and read them again for every
# token, which is what made decoding 8x slower on the development VM. Copying
# the weights into RAM avoids that, but only when they fit with room to spare.
choose_load_mode() {
  if [[ "$LOAD_MODE" != auto ]]; then
    echo "$LOAD_MODE"
    return
  fi
  if ((GPU_LAYERS > 0)); then
    echo mmap
    return
  fi
  local need
  need=$(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$MODEL_DIR/$MODEL_FILE")
  if [[ -n "$MMPROJ_FILE" ]]; then
    need=$((need + $(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$MODEL_DIR/$MMPROJ_FILE")))
  fi
  need=$((need + 1024 * 1024 * 1024))
  local available
  available=$(available_bytes)
  log "weights + 1 GiB need $((need >> 20)) MiB; $((available >> 20)) MiB free"
  if ((need < available)); then
    echo none
  else
    log "weights + 1 GiB do not fit in free memory; using mmap (slower decoding)"
    echo mmap
  fi
}

server_pid=""

stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}

start_server() {
  local log_file="$OUT_DIR/llama-server.log" load_mode
  local -a extra=()
  mkdir -p "$OUT_DIR"
  load_mode=$(choose_load_mode)
  if [[ -n "$MMPROJ_FILE" ]]; then
    extra+=(--mmproj "$MODEL_DIR/$MMPROJ_FILE")
  fi
  # Repacking copies the CPU weights into private memory (1.4 GiB for 4B
  # Q4_K_M), which defeats mmap and got llama-server killed under a 3 GiB limit.
  if [[ "$load_mode" == mmap ]]; then
    extra+=(--no-repack)
  fi
  LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    nice -n "$NICE_LEVEL" "$LLAMA_DIR/llama-server" \
    --model "$MODEL_DIR/$MODEL_FILE" \
    "${extra[@]}" \
    --load-mode "$load_mode" \
    --threads "$THREADS" \
    --n-gpu-layers "$GPU_LAYERS" \
    --ctx-size 4096 \
    --parallel 1 \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$log_file" 2>&1 &
  server_pid=$!
  trap stop_server EXIT
  trap 'exit 143' TERM INT

  local waited=0
  until curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "llama-server exited; last lines of $log_file:"
      tail -n 40 "$log_file" >&2
      exit 1
    fi
    if ((waited >= 600)); then
      log "llama-server did not become healthy within 600 s"
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  log "llama-server healthy after ${waited}s (threads=$THREADS," \
    "gpu layers=$GPU_LAYERS, load mode=$load_mode)"
}

run() {
  start_server
  local started remaining task tokens
  started=$(date +%s)
  for task in $TASKS; do
    remaining=$((TIME_BUDGET - ($(date +%s) - started)))
    if ((remaining <= 0)); then
      log "time budget used up; skipping $task and the tasks after it"
      return 1
    fi
    tokens=$MAX_TOKENS
    [[ "$task" == ocr-history ]] && tokens=$PAGE_MAX_TOKENS
    log "task $task: $COUNT items, ${remaining}s left"
    python -m medbench.cli run \
      --task "$task" \
      --backend openai \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --model-name "$MODEL_FILE" \
      --count "$COUNT" \
      --max-tokens "$tokens" \
      --time-budget "$remaining" \
      --request-timeout "$remaining" \
      --out "$OUT_DIR/$task.json"
  done
  if [[ "${RUN_PYTEST:-1}" == 1 ]]; then
    log "pytest smoke tests against the server"
    MEDBENCH_BASE_URL="http://127.0.0.1:$PORT/v1" \
      MEDBENCH_MODEL="$MODEL_FILE" \
      MEDBENCH_VISION="$([[ -n "$MMPROJ_FILE" ]] && echo 1 || echo 0)" \
      python -m pytest tests/test_real_models.py -m model -v -p no:cacheprovider \
      --junitxml "$OUT_DIR/real-models.xml"
  fi
  grep -E "prompt eval time|eval time|image slice encoded|encoding image" \
    "$OUT_DIR/llama-server.log" >"$OUT_DIR/llama-timings.txt" || true
}

case "${1:-}" in
  download|run)
    if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
      log "This pinned llama.cpp build requires Linux x86_64; see docs/apple-silicon.md for MLX."
      exit 2
    fi
    "${1}"
    ;;
  preflight) choose_load_mode ;;
  *)
    echo "Usage: $0 download|run|preflight" >&2
    exit 2
    ;;
esac
