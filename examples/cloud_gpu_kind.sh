#!/usr/bin/env bash
# End-to-end run of the Kubernetes Job flow on a local kind cluster (issue #23).
#
# No GPU and no cloud account: the Job asks for 0 GPUs, UniMedVL uses its
# echo engine, and everything else is the real path - the rendered Job,
# streaming the commit into the pod, scripts/gpu-host-run.sh installing uv
# and medbench, the run, collecting the reports and deleting the Job.
#
#   bash examples/cloud_gpu_kind.sh [log file]
#
# Needs docker, kind and kubectl. Uses the kind cluster $CLUSTER if it
# exists (the CI job creates it with helm/kind-action), else creates it and
# deletes it afterwards unless KEEP_CLUSTER=1. The same command with
# --kube-context of a real GPU cluster and without --gpu-count 0,
# --no-require-gpu and the echo engine is a real GPU test.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CLUSTER="${CLUSTER:-medbench-issue23}"
WORKDIR="${WORKDIR:-.cache/cloud-gpu/kind-experiment}"
LOG="${1:-$WORKDIR.log}"

# shellcheck disable=SC2329 # called by the EXIT trap
cleanup() {
  if [[ "${KEEP_CLUSTER:-0}" != 1 ]]; then
    kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true
  fi
}

if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  trap cleanup EXIT
  kind create cluster --name "$CLUSTER" --wait 120s
fi
rm -rf "$WORKDIR"
mkdir -p "$(dirname "$LOG")"
status=0
python3 -m scripts.cloud_gpu all \
  --provider kubernetes --mode k8s --kube-context "kind-$CLUSTER" \
  --model unimedvl --gpu-count 0 --no-require-gpu \
  --steps run --count 2 --tasks diagnosis --time-budget 600 --ttl 60 \
  --env UNIMEDVL_ENGINE=echo --env RUN_PYTEST=0 \
  --workdir "$WORKDIR" --yes 2>&1 | tee "$LOG" || status=$?
echo "exit code: $status" | tee -a "$LOG"
find "$WORKDIR/reports" -maxdepth 2 -type f 2>/dev/null | sort | tee -a "$LOG"
for file in "$WORKDIR"/reports/gpu-host-run.json "$WORKDIR"/reports/*/diagnosis.json; do
  if [[ -f "$file" ]]; then
    { head -c 600 "$file"; echo; } | tee -a "$LOG"
  fi
done
kubectl --context "kind-$CLUSTER" get jobs --all-namespaces | tee -a "$LOG"
exit "$status"
