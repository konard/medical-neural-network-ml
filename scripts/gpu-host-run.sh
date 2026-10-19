#!/usr/bin/env bash
# Run the GPU model tests on the machine this script runs on (issue #23).
#
# One entry point for every place a GPU test can run: a cloud VM reached over
# SSH, a Kubernetes Job, an ephemeral GitHub Actions runner, a GitLab job or a
# GPU host we own. Run it from the repository root:
#
#   GPU_MODEL=medgemma-27b-it bash scripts/gpu-host-run.sh
#
# It checks the NVIDIA driver, installs a pinned uv, creates a virtual
# environment with medbench, then downloads and runs the model with
# scripts/medbench-model-run.sh. Downloads and runs go through
# `medbench monitor`, so the reports hold peak GPU memory, driver and wall
# time. The exit code is the exit code of the model run.
#
# Environment:
#   GPU_MODEL        medgemma-27b-it, medgemma-4b-it or unimedvl (medgemma-4b-it)
#   COUNT            items per task (20)
#   TASKS            space-separated medbench tasks (all seven GPU tasks)
#   TIME_BUDGET      seconds for all tasks together (5400)
#   OUT_DIR          reports (medbench-reports)
#   CACHE_DIR        downloads, uv and virtual environments (.cache/medbench)
#   REQUIRE_GPU      1 fails without an NVIDIA driver of at least MIN_DRIVER (1)
#   MIN_DRIVER       oldest driver branch the CUDA 13.4 build supports (580)
#   MEDBENCH_PYTHON  use this interpreter as is (no uv, no install); for tests
#                    and hosts that already have medbench installed
#   PYTHON_VERSION   Python for medbench, installed by uv (3.15)
#   STEPS            download and/or run (download run); the echo engine of
#                    UniMedVL needs no download
#   CLEANUP          1 removes CACHE_DIR afterwards, for a GPU host that is
#                    reused (0); reports stay
#   MEDBENCH_COMMIT  commit being tested, recorded in the summary
# The model's own settings come from scripts/gpu-model-env.sh; any variable
# of scripts/medbench-model-run.sh can still be overridden.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

GPU_MODEL="${GPU_MODEL:-medgemma-4b-it}"
export COUNT="${COUNT:-20}"
export TASKS="${TASKS:-diagnosis diagnosis-ddxplus ocr-prescription ocr-history imaging-xray imaging-mri imaging-ultrasound}"
export TIME_BUDGET="${TIME_BUDGET:-5400}"
export OUT_DIR="${OUT_DIR:-medbench-reports}"
export CACHE_DIR="${CACHE_DIR:-.cache/medbench}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
MIN_DRIVER="${MIN_DRIVER:-580}"
PYTHON_VERSION="${PYTHON_VERSION:-3.15}"
UNIMEDVL_PYTHON_VERSION="3.14"  # scripts/unimedvl-requirements.in
CLEANUP="${CLEANUP:-0}"
STEPS="${STEPS:-download run}"
UV_VERSION="0.13.1"
UV_SHA256_X86_64="eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220"
UV_SHA256_AARCH64="9041c73c99fdc5892cf128140beb354eca9231d502068cc36e6cb53a8e59ad14"

log() {
  printf '[gpu-host-run] %s\n' "$*" >&2
}

# driver_major "580.95.05" -> 580
driver_major() {
  local version="$1"
  printf '%s\n' "${version%%.*}"
}

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found: install NVIDIA driver $MIN_DRIVER or newer" \
      "(infra/ansible/gpu-host.yml) or set REQUIRE_GPU=0"
    return 1
  fi
  nvidia-smi
  local version major
  version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
  major="$(driver_major "$version")"
  if [[ ! "$major" =~ ^[0-9]+$ ]] || ((major < MIN_DRIVER)); then
    log "NVIDIA driver $version is older than $MIN_DRIVER," \
      "which the CUDA 13.4 build of llama.cpp needs"
    return 1
  fi
  log "NVIDIA driver $version"
}

install_uv() {
  local arch sha dir archive
  arch="$(uname -m)"
  case "$arch" in
    x86_64) sha="$UV_SHA256_X86_64" ;;
    aarch64) sha="$UV_SHA256_AARCH64" ;;
    *) log "no pinned uv for $arch"; return 1 ;;
  esac
  dir="$CACHE_DIR/uv-$UV_VERSION"
  if [[ ! -x "$dir/uv" ]]; then
    archive="$CACHE_DIR/uv-$UV_VERSION-$arch.tar.gz"
    mkdir -p "$dir"
    curl --fail --location --retry 3 --silent --show-error --output "$archive" \
      "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$arch-unknown-linux-gnu.tar.gz"
    if ! printf '%s  %s\n' "$sha" "$archive" | sha256sum --check --status; then
      log "SHA-256 mismatch for $archive"
      return 1
    fi
    tar --extract --gzip --file "$archive" --directory "$dir" --strip-components 1
    rm -f "$archive"
  fi
  UV="$dir/uv"
  export UV_PYTHON_INSTALL_DIR="$CACHE_DIR/python"
  export UV_CACHE_DIR="$CACHE_DIR/uv-cache"
}

setup_python() {
  if [[ -n "${MEDBENCH_PYTHON:-}" ]]; then
    log "using $MEDBENCH_PYTHON as is"
    PATH="$(dirname "$MEDBENCH_PYTHON"):$PATH"
    return
  fi
  install_uv
  local venv="$CACHE_DIR/medbench-venv"
  "$UV" venv --quiet --allow-existing --python "$PYTHON_VERSION" "$venv"
  "$UV" pip install --quiet --python "$venv/bin/python" -e ".[dev]"
  if [[ "$venv" != /* ]]; then
    venv="$ROOT/$venv"
  fi
  PATH="$venv/bin:$PATH"
  if [[ "$GPU_MODEL" == unimedvl && -z "${UNIMEDVL_PYTHON:-}" ]]; then
    "$UV" python install --quiet "$UNIMEDVL_PYTHON_VERSION"
    UNIMEDVL_PYTHON="$("$UV" python find "$UNIMEDVL_PYTHON_VERSION")"
    export UNIMEDVL_PYTHON
  fi
}

# Export the model profile, keeping anything the caller already set.
load_model() {
  local line
  local -a profile
  mapfile -t profile < <(bash "$SCRIPT_DIR/gpu-model-env.sh" "$GPU_MODEL")
  for line in "${profile[@]}"; do
    local key="${line%%=*}"
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${line#*=}"
    fi
  done
}

summary() {
  local status="$1" started="$2"
  python3 - "$OUT_DIR/gpu-host-run.json" "$GPU_MODEL" "$status" "$started" <<'PY'
import json, os, sys, time
path, model, status, started = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "model": model,
            "exit_code": int(status),
            "commit": os.environ.get("MEDBENCH_COMMIT", ""),
            "count": int(os.environ["COUNT"]),
            "tasks": os.environ["TASKS"].split(),
            "wall_seconds": round(time.time() - float(started), 1),
        },
        handle,
        indent=2,
    )
    handle.write("\n")
PY
}

main() {
  local started status=0
  started="$(date +%s)"
  mkdir -p "$OUT_DIR" "$CACHE_DIR"
  bash "$SCRIPT_DIR/gpu-model-env.sh" "$GPU_MODEL" >/dev/null
  if [[ "$REQUIRE_GPU" == 1 ]]; then
    check_gpu
  fi
  setup_python
  load_model
  log "model $GPU_MODEL, $COUNT items per task, tasks: $TASKS"
  if [[ " $STEPS " == *" download "* ]]; then
    python -m medbench.cli monitor --out "$OUT_DIR/resources-download.json" \
      --label "Download model" -- bash "$SCRIPT_DIR/medbench-model-run.sh" download \
      || status=$?
  fi
  if ((status == 0)) && [[ " $STEPS " == *" run "* ]]; then
    python -m medbench.cli monitor --out "$OUT_DIR/resources-run.json" \
      --label "Run medbench" -- bash "$SCRIPT_DIR/medbench-model-run.sh" run \
      || status=$?
  fi
  summary "$status" "$started"
  if [[ "$CLEANUP" == 1 ]]; then
    rm -rf "$CACHE_DIR"
  fi
  log "finished with exit code $status"
  return "$status"
}

main "$@"
