#!/usr/bin/env bash
# Start one ephemeral GitHub Actions runner on a temporary GPU host (issue #23).
#
# `python -m scripts.cloud_gpu up --mode runner` runs this as root over SSH
# and writes the runner's just-in-time (JIT) configuration to stdin, so the
# secret is never on a command line, in Terraform state or in cloud user
# data. The script installs the pinned runner, keeps the configuration in a
# root-only file and starts a systemd unit that runs exactly one job as the
# unprivileged `runner` user. The runner reads ACTIONS_RUNNER_INPUT_JITCONFIG
# and removes it from its environment. With --poweroff the host powers off
# after the job (cloud VMs are then deleted by `down` or the reaper).
#
# Usage (as root):
#   printf '%s' "$ENCODED_JIT_CONFIG" | bash scripts/gpu-runner-start.sh [--poweroff]
#   bash scripts/gpu-runner-start.sh serve [--poweroff]   # used by the unit
set -euo pipefail

# The same runner as infra/ansible/github-runner.yml and
# infra/kubernetes/arc-gpu-runner-values.yaml
# (dev/log/issues/25/pulls/26/data/runner-pins.json).
RUNNER_VERSION="2.338.2"
RUNNER_SHA256="baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40"
RUNNER_HOME="${RUNNER_HOME:-/opt/actions-runner}"
RUNNER_USER="${RUNNER_USER:-runner}"
CONFIG_DIR=/run/medbench-runner
CONFIG_FILE="$CONFIG_DIR/jit-config"
UNIT=medbench-jit-runner
SERVE=/usr/local/sbin/medbench-jit-runner

log() {
  printf '[gpu-runner-start] %s\n' "$*" >&2
}

install_runner() {
  local tmp
  if [[ -x "$RUNNER_HOME/run.sh" ]] && [[ -f "$RUNNER_HOME/.medbench-$RUNNER_VERSION" ]]; then
    return 0
  fi
  if [[ "$(uname -m)" != x86_64 ]]; then
    log "the runner download is pinned for x86_64 only"
    return 1
  fi
  id -u "$RUNNER_USER" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$RUNNER_USER"
  tmp="$(mktemp -d)"
  curl --fail --location --retry 3 --silent --show-error --output "$tmp/runner.tar.gz" \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  if ! printf '%s  %s\n' "$RUNNER_SHA256" "$tmp/runner.tar.gz" | sha256sum --check --status; then
    log "SHA-256 mismatch for the runner download"
    rm -rf "$tmp"
    return 1
  fi
  mkdir -p "$RUNNER_HOME"
  tar -xzf "$tmp/runner.tar.gz" -C "$RUNNER_HOME"
  rm -rf "$tmp"
  "$RUNNER_HOME/bin/installdependencies.sh" >/dev/null
  chown -R "$RUNNER_USER:" "$RUNNER_HOME"
  touch "$RUNNER_HOME/.medbench-$RUNNER_VERSION"
  log "runner $RUNNER_VERSION installed in $RUNNER_HOME"
}

# Runs inside the systemd unit, as root: hands the configuration to the
# runner through its environment and deletes the file first.
serve() {
  local config status=0
  config="$(cat "$CONFIG_FILE")"
  rm -f "$CONFIG_FILE"
  rm -rf "${RUNNER_HOME:?}/_work"
  ACTIONS_RUNNER_INPUT_JITCONFIG="$config" runuser -u "$RUNNER_USER" \
    -w ACTIONS_RUNNER_INPUT_JITCONFIG -- "$RUNNER_HOME/run.sh" || status=$?
  log "the runner finished with exit code $status"
  if [[ "${1:-}" == --poweroff ]]; then
    log "powering off"
    shutdown -P now "medbench: ephemeral runner done"
  fi
  return "$status"
}

start() {
  local config
  config="$(cat)"
  if [[ -z "$config" ]]; then
    log "expected the encoded JIT configuration on stdin"
    return 2
  fi
  install_runner
  install -d -m 0700 "$CONFIG_DIR"
  (umask 077 && printf '%s' "$config" >"$CONFIG_FILE")
  install -m 0755 "${BASH_SOURCE[0]}" "$SERVE"
  systemctl reset-failed "$UNIT" >/dev/null 2>&1 || true
  systemd-run --unit="$UNIT" --collect --description="medbench ephemeral GPU runner" \
    --setenv=RUNNER_HOME="$RUNNER_HOME" --setenv=RUNNER_USER="$RUNNER_USER" \
    "$SERVE" serve "$@"
  log "runner unit $UNIT started; follow it with: journalctl -fu $UNIT"
}

main() {
  if ((EUID != 0)); then
    log "run as root"
    return 1
  fi
  if [[ "${1:-}" == serve ]]; then
    shift
    serve "$@"
  else
    start "$@"
  fi
}

main "$@"
