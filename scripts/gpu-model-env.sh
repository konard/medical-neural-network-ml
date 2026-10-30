#!/usr/bin/env bash
# Print the pinned settings of one GPU model profile as KEY=VALUE lines
# (issue #23), for scripts/gpu-host-run.sh, the Kubernetes Job and any CI
# system that wants to run the GPU tests without GitHub Actions.
#
# Usage:
#   bash scripts/gpu-model-env.sh medgemma-27b-it   # or medgemma-4b-it, unimedvl
#
# The values are the ones the `gpu` job of .github/workflows/model-tests.yml
# uses; tests/test_cloud_gpu_scripts.py keeps the two in step. Every download
# is pinned by a commit or a SHA-256, and scripts/medbench-model-run.sh checks
# them before anything runs.
set -euo pipefail

# llama.cpp CUDA 13.4 build: needs NVIDIA driver R580 or newer and a Turing
# (compute capability 7.5) or newer GPU.
gpu_llama() {
  echo "LLAMA_ARCHIVE=llama-b12012-bin-ubuntu-cuda-13.4-x64.tar.gz"
  echo "LLAMA_SHA256=324ee513196aa1fa72627ed978276e3fa117cc6f7cf6cf5ff7c5218ef70a0345"
  echo "CUDART_ARCHIVE=cudart-llama-b12012-bin-ubuntu-cuda-13.4-x64.tar.gz"
  echo "CUDART_SHA256=01e16c460a2b9bd142ca040ced8b34e0ae74d5c0220c306e8a7ac2c25565b492"
  echo "GPU_LAYERS=999"
}

case "${1:-}" in
  medgemma-27b-it)
    gpu_llama
    echo "MODEL_REPO=unsloth/medgemma-27b-it-GGUF"
    echo "MODEL_REVISION=c56c561dd4fcb1b57bc64cd75cc8f728eb5befcd"
    echo "MODEL_FILE=medgemma-27b-it-Q4_K_M.gguf"
    echo "MODEL_SHA256=89fdb77fd378f4e4f7f1ed2bb349c8cb0784f354970736758368f9d3cc7449c3"
    echo "MMPROJ_FILE=mmproj-F16.gguf"
    echo "MMPROJ_SHA256=9c7733e3ced596d4e05b890ca66622fed129716c4a23e7feff66528416a5b0e0"
    ;;
  medgemma-4b-it)
    # The model itself is the default of scripts/medbench-model-run.sh.
    gpu_llama
    ;;
  unimedvl)
    echo "SERVER=unimedvl"
    ;;
  *)
    echo "gpu-model-env: unknown model '${1:-}'" \
      "(medgemma-27b-it, medgemma-4b-it, unimedvl)" >&2
    exit 2
    ;;
esac
