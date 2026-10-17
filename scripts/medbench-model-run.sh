#!/usr/bin/env bash
# Run the medical benchmark (real openly licensed samples) against a real model (issue #1).
#
# Usage:
#   bash scripts/medbench-model-run.sh download   # fetch and verify files
#   bash scripts/medbench-model-run.sh run        # serve the model, run tasks
#   bash scripts/medbench-model-run.sh preflight  # print the chosen load mode
#
# Every download is pinned to a SHA-256 (or a Git commit), so a changed file
# fails instead of being run. The model is served by an OpenAI-compatible
# server and medbench talks to it over HTTP:
#   SERVER=llama     llama.cpp's llama-server with GGUF files (default); works
#                    on a Linux CPU runner or an NVIDIA GPU runner.
#   SERVER=unimedvl  scripts/unimedvl_server.py with upstream UniMedVL code
#                    and bf16 weights (issue #6); NVIDIA GPU only.
# On Apple Silicon use the native MLX backend described in docs/apple-silicon.md.
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
#   HEALTH_TIMEOUT   seconds to wait for the server to load the model (600;
#                    1800 for UniMedVL, which reads 29 GB of bf16 weights)
#
# UniMedVL only (SERVER=unimedvl):
#   UNIMEDVL_PYTHON      Python 3.13 used to create the model's own venv (python3.13)
#   UNIMEDVL_VENV        that venv ($CACHE_DIR/unimedvl-venv)
#   UNIMEDVL_ENGINE      unimedvl, or echo to test the plumbing without a GPU
#   UNIMEDVL_MAX_GPU_MEMORY  weights allowed on GPU 0, the rest stays on the CPU (40GiB)
set -euo pipefail

LLAMA_VERSION="${LLAMA_VERSION:-b11707}"
LLAMA_ARCHIVE="${LLAMA_ARCHIVE:-llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz}"
LLAMA_SHA256="${LLAMA_SHA256:-88879871f765c3dec22260046e10a76e2ed38de5bdf4a7f68049245a8f4056c8}"
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

SERVER="${SERVER:-llama}"

# UniMedVL: upstream code and weights, both Apache-2.0 (THIRD_PARTY_NOTICES.md).
UNIMEDVL_CODE_REPO="https://github.com/uni-medical/UniMedVL"
UNIMEDVL_CODE_COMMIT="f51aa0accbd02e846e5ea20b595fb39b516dbe1f"
UNIMEDVL_MODEL_REPO="General-Medical-AI/UniMedVL"
UNIMEDVL_MODEL_REVISION="29918cef3ce94a13fe1670145c32a3d1198aec7d"
# File and SHA-256 at that revision (ae.safetensors, the image decoder, is
# not needed for answering questions and is not downloaded).
UNIMEDVL_FILES=(
  "ema_bf16.safetensors 254873338061b90aefa2ec89fa07e4a50ede812c84dd72a83a13fd7db2237b18"
  "config.json 468f9050c5bbc3d319bc0501c3dadf33a797240adbf623739686b5b3fc160d5a"
  "llm_config.json 565af3cc3dd5b478edaa40d3a6b448b489da820d18594d58f5b902ffc8149cdb"
  "vit_config.json 020814af27b704b29a676cd86f8514ac3075b15dac2dc88f9cc430ed9e1bef6b"
  "tokenizer.json c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
  "tokenizer_config.json 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
  "vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
  "merges.txt 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
)
# Upstream imports flash_attn_varlen_func; this prebuilt wheel matches the
# pinned torch 2.8.0 (CUDA 12, the C++11 ABI that PyPI torch wheels use
# since 2.7, and CPython 3.13 like UNIMEDVL_PYTHON). v2.8.3.post2 also fixes GHSA-7g5w-pq96-8c5w (issue #19).
FLASH_ATTN_WHEEL="flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/${FLASH_ATTN_WHEEL//+/%2B}"
FLASH_ATTN_SHA256="9d5bcc61b3251b0a9a9ccea6dc75fd002e0fcaaf2edc48df7d981bb114f5b4c7"
UNIMEDVL_PYTHON="${UNIMEDVL_PYTHON:-python3.13}"
UNIMEDVL_VENV="${UNIMEDVL_VENV:-$CACHE_DIR/unimedvl-venv}"
UNIMEDVL_ENGINE="${UNIMEDVL_ENGINE:-unimedvl}"
UNIMEDVL_MAX_GPU_MEMORY="${UNIMEDVL_MAX_GPU_MEMORY:-40GiB}"
UNIMEDVL_CODE_DIR="$CACHE_DIR/unimedvl-code-${UNIMEDVL_CODE_COMMIT:0:12}"
UNIMEDVL_MODEL_DIR="$CACHE_DIR/models/unimedvl-${UNIMEDVL_MODEL_REVISION:0:12}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$SERVER" == unimedvl ]]; then
  HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
  SERVED_MODEL=unimedvl
else
  HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
  SERVED_MODEL="$MODEL_FILE"
fi

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

download_llama() {
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

# Upstream code at a fixed commit, its weights by SHA-256, and a separate
# Python 3.13 venv with upstream's pinned packages (medbench itself does not
# import torch). Nothing from the checkout runs until the server starts.
download_unimedvl() {
  local hub="https://huggingface.co/$UNIMEDVL_MODEL_REPO/resolve/$UNIMEDVL_MODEL_REVISION"
  local entry name sha
  if [[ ! -d "$UNIMEDVL_CODE_DIR/.git" ]]; then
    log "cloning $UNIMEDVL_CODE_REPO at $UNIMEDVL_CODE_COMMIT"
    git init --quiet "$UNIMEDVL_CODE_DIR"
    git -C "$UNIMEDVL_CODE_DIR" fetch --quiet --depth 1 \
      "$UNIMEDVL_CODE_REPO" "$UNIMEDVL_CODE_COMMIT"
    git -C "$UNIMEDVL_CODE_DIR" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
  fi
  if [[ "$(git -C "$UNIMEDVL_CODE_DIR" rev-parse HEAD)" != "$UNIMEDVL_CODE_COMMIT" ]]; then
    log "UniMedVL checkout is not at $UNIMEDVL_CODE_COMMIT; delete $UNIMEDVL_CODE_DIR"
    exit 1
  fi
  log "verified UniMedVL code at $UNIMEDVL_CODE_COMMIT"
  for entry in "${UNIMEDVL_FILES[@]}"; do
    read -r name sha <<<"$entry"
    fetch "$hub/$name" "$UNIMEDVL_MODEL_DIR/$name" "$sha"
  done
  fetch "$FLASH_ATTN_URL" "$CACHE_DIR/$FLASH_ATTN_WHEEL" "$FLASH_ATTN_SHA256"
  if [[ ! -x "$UNIMEDVL_VENV/bin/python" ]]; then
    "$UNIMEDVL_PYTHON" -m venv "$UNIMEDVL_VENV"
  fi
  "$UNIMEDVL_VENV/bin/python" -m pip install --quiet --upgrade pip
  "$UNIMEDVL_VENV/bin/python" -m pip install --quiet \
    -r "$SCRIPT_DIR/unimedvl-requirements.txt"
  "$UNIMEDVL_VENV/bin/python" -m pip install --quiet --no-deps \
    "$CACHE_DIR/$FLASH_ATTN_WHEEL"
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

launch_unimedvl() {
  local log_file="$1" code_dir="$UNIMEDVL_CODE_DIR" python="$UNIMEDVL_VENV/bin/python"
  if [[ "$UNIMEDVL_ENGINE" == echo ]]; then
    # Plumbing test: no checkout, weights or GPU needed.
    code_dir="$SCRIPT_DIR"
    python="${UNIMEDVL_ECHO_PYTHON:-python3}"
  fi
  # -I: the checkout's folder is not put on sys.path implicitly; the server
  # adds codes/ itself, after changing into the checkout. Offline mode: the
  # weights and configs are already verified on disk, so transformers must
  # never fetch code or files from the Hub (see the accepted advisories in
  # scripts/audit_dependencies.py).
  (cd "$code_dir" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    exec "$python" -I "$SCRIPT_DIR/unimedvl_server.py" \
    --engine "$UNIMEDVL_ENGINE" \
    --code-dir "$code_dir" \
    --model-dir "$UNIMEDVL_MODEL_DIR" \
    --max-gpu-memory "$UNIMEDVL_MAX_GPU_MEMORY" \
    --port "$PORT") >"$log_file" 2>&1 &
  server_pid=$!
}

launch_llama() {
  local log_file="$1" load_mode
  local -a extra=()
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
    --metrics \
    >"$log_file" 2>&1 &
  server_pid=$!
  log "llama-server: threads=$THREADS, gpu layers=$GPU_LAYERS, load mode=$load_mode"
}

server_log() {
  if [[ "$SERVER" == unimedvl ]]; then
    echo "$OUT_DIR/unimedvl-server.log"
  else
    echo "$OUT_DIR/llama-server.log"
  fi
}

start_server() {
  local log_file
  log_file=$(server_log)
  mkdir -p "$OUT_DIR"
  case "$SERVER" in
    llama) launch_llama "$log_file" ;;
    unimedvl) launch_unimedvl "$log_file" ;;
    *)
      log "unknown SERVER '$SERVER' (llama or unimedvl)"
      exit 2
      ;;
  esac
  # The server is stopped on every exit (done, failed or out of time), so a
  # self-hosted runner does not keep a model in GPU memory between jobs.
  trap stop_server EXIT
  trap 'exit 143' TERM INT

  local started waited=0
  started=$(date +%s)
  until curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "$SERVER server exited; last lines of $log_file:"
      tail -n 40 "$log_file" >&2
      exit 1
    fi
    waited=$(($(date +%s) - started))
    if ((waited >= HEALTH_TIMEOUT)); then
      log "$SERVER server did not become healthy within $HEALTH_TIMEOUT s"
      exit 1
    fi
    sleep 2
  done
  waited=$(($(date +%s) - started))
  printf '{"server": "%s", "cold_start_seconds": %d}\n' "$SERVER" "$waited" \
    >"$OUT_DIR/server-start.json"
  log "$SERVER server healthy after ${waited}s"
}

# Save what the server knows about itself: versions, limits, peak GPU
# memory and speed (UniMedVL JSON) or llama.cpp's Prometheus counters plus
# its /props (context size, build, chat template).
save_server_metrics() {
  if [[ "$SERVER" == unimedvl ]]; then
    curl --silent --fail "http://127.0.0.1:$PORT/metrics" \
      >"$OUT_DIR/server-metrics.json" || true
  else
    curl --silent --fail "http://127.0.0.1:$PORT/metrics" \
      >"$OUT_DIR/llama-metrics.txt" || true
    curl --silent --fail "http://127.0.0.1:$PORT/props" \
      >"$OUT_DIR/llama-props.json" || true
  fi
}

run() {
  start_server
  local started remaining task tokens
  started=$(date +%s)
  for task in $TASKS; do
    remaining=$((TIME_BUDGET - ($(date +%s) - started)))
    if ((remaining <= 0)); then
      log "time budget used up; skipping $task and the tasks after it"
      save_server_metrics
      return 1
    fi
    tokens=$MAX_TOKENS
    [[ "$task" == ocr-history ]] && tokens=$PAGE_MAX_TOKENS
    log "task $task: $COUNT items, ${remaining}s left"
    python -m medbench.cli run \
      --task "$task" \
      --backend openai \
      --base-url "http://127.0.0.1:$PORT/v1" \
      --model-name "$SERVED_MODEL" \
      --count "$COUNT" \
      --max-tokens "$tokens" \
      --time-budget "$remaining" \
      --request-timeout "$remaining" \
      --out "$OUT_DIR/$task.json"
  done
  if [[ "${RUN_PYTEST:-1}" == 1 ]]; then
    log "pytest smoke tests against the server"
    MEDBENCH_BASE_URL="http://127.0.0.1:$PORT/v1" \
      MEDBENCH_MODEL="$SERVED_MODEL" \
      MEDBENCH_VISION="$([[ "$SERVER" == unimedvl || -n "$MMPROJ_FILE" ]] && echo 1 || echo 0)" \
      python -m pytest tests/test_real_models.py -m model -v -p no:cacheprovider \
      --junitxml "$OUT_DIR/real-models.xml"
  fi
  save_server_metrics
  if [[ "$SERVER" == llama ]]; then
    grep -E "prompt eval time|eval time|image slice encoded|encoding image" \
      "$OUT_DIR/llama-server.log" >"$OUT_DIR/llama-timings.txt" || true
  fi
}

download() {
  if [[ "$SERVER" == unimedvl ]]; then
    download_unimedvl
  else
    download_llama
  fi
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
