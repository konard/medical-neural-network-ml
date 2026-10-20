#!/usr/bin/env bash
# One ephemeral GitHub Actions runner after another on a GPU host we keep
# (issue #23), installed by infra/ansible/github-runner.yml as a systemd
# service. Each round asks GitHub for a just-in-time (JIT) runner config,
# runs exactly one job with it and wipes the work directory, so no job sees
# files of the job before it. The JIT config goes to the runner through
# ACTIONS_RUNNER_INPUT_JITCONFIG, which the runner reads and removes from
# its environment, so it never shows up in a process list.
#
# /etc/medbench-runner/runner.env (root only) holds:
#   GITHUB_REPOSITORY  owner/repo
#   RUNNER_TOKEN       fine-grained token, "Administration: read and write"
#   RUNNER_LABELS      comma-separated labels (self-hosted,linux,x64,gpu)
#   RUNNER_GROUP_ID    runner group (1, the default group)
set -euo pipefail

RUNNER_HOME="${RUNNER_HOME:-/opt/actions-runner}"
RUNNER_USER="${RUNNER_USER:-runner}"
API="${GITHUB_API_URL:-https://api.github.com}"

log() {
  printf '[medbench-runner] %s\n' "$*" >&2
}

jit_config() {
  local name="$1"
  python3 - "$name" <<'PY'
import json, os, sys, urllib.request

body = json.dumps({
    "name": sys.argv[1],
    "runner_group_id": int(os.environ.get("RUNNER_GROUP_ID", "1")),
    "labels": os.environ.get("RUNNER_LABELS", "self-hosted,linux,x64,gpu").split(","),
    "work_folder": "_work",
}).encode()
request = urllib.request.Request(
    f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}"
    f"/repos/{os.environ['GITHUB_REPOSITORY']}/actions/runners/generate-jitconfig",
    data=body,
    method="POST",
    headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['RUNNER_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28",
    },
)
with urllib.request.urlopen(request, timeout=60) as response:
    print(json.load(response)["encoded_jit_config"])
PY
}

main() {
  local name config status
  : "${GITHUB_REPOSITORY:?set in /etc/medbench-runner/runner.env}"
  : "${RUNNER_TOKEN:?set in /etc/medbench-runner/runner.env}"
  log "using $API for $GITHUB_REPOSITORY"
  while true; do
    name="$(hostname -s)-$(date +%s)"
    if ! config="$(jit_config "$name")"; then
      log "could not get a JIT config; retrying in 60 seconds"
      sleep 60
      continue
    fi
    rm -rf "${RUNNER_HOME:?}/_work"
    log "runner $name waits for one job"
    status=0
    # sudo resets the environment, so the job never sees RUNNER_TOKEN; only
    # the JIT config is handed over, through the environment, not argv.
    ACTIONS_RUNNER_INPUT_JITCONFIG="$config" sudo --set-home --user "$RUNNER_USER" \
      --preserve-env=ACTIONS_RUNNER_INPUT_JITCONFIG -- "$RUNNER_HOME/run.sh" || status=$?
    log "runner $name finished with exit code $status"
    rm -rf "${RUNNER_HOME:?}/_work"
  done
}

main "$@"
