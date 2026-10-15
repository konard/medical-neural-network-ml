#!/usr/bin/env bash
#
# Check that the repository's Dependency graph is enabled.
#
# actions/dependency-review-action fails every pull request with "Dependency
# review is not supported on this repository" when the Dependency graph is
# switched off in Settings -> Code security. That is a repository setting,
# not a defect in the pull request, so:
#
# DEPENDENCY_GRAPH_MODE:
#   report  -- pull requests. A disabled graph is a warning, and
#              `enabled=false` is written to $GITHUB_OUTPUT so the review
#              step is skipped instead of failing.
#   require -- push to main and scheduled runs. A disabled graph fails the
#              run with the setting a maintainer has to change.
#
# Only a definite 404 from the SBOM endpoint means "disabled". Any other
# answer (rate limit, timeout, outage) has not said the setting is off, so
# the review step still runs and reports for itself.

set -u

MODE="${DEPENDENCY_GRAPH_MODE:-report}"
API="${GITHUB_API_URL:-https://api.github.com}"
SERVER="${GITHUB_SERVER_URL:-https://github.com}"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
CURL_TIMEOUT="${DEPENDENCY_GRAPH_CURL_TIMEOUT:-20}"
SETTINGS_URL="${SERVER}/${REPO}/settings/security_analysis"

case "$MODE" in
  report | require) ;;
  *)
    echo "::error::DEPENDENCY_GRAPH_MODE must be report or require, got '$MODE'"
    exit 2
    ;;
esac

set_output() {
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf 'enabled=%s\n' "$1" >>"$GITHUB_OUTPUT"
  fi
  printf 'enabled=%s\n' "$1"
}

status=$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --max-time "$CURL_TIMEOUT" \
  -H "Accept: application/vnd.github+json" \
  ${GITHUB_TOKEN:+-H "Authorization: Bearer $GITHUB_TOKEN"} \
  "$API/repos/$REPO/dependency-graph/sbom")
status=${status:-000}

case "$status" in
  200)
    echo "Dependency graph is enabled for $REPO."
    set_output true
    ;;
  404)
    message="Dependency graph is disabled for $REPO, so dependency review cannot run. A repository admin must enable it: $SETTINGS_URL -> Dependency graph -> Enable."
    if [ "$MODE" = require ]; then
      echo "::error title=Dependency graph disabled::$message"
      set_output false
      exit 1
    fi
    echo "::warning title=Dependency graph disabled::$message Skipping dependency review on this pull request; main fails until it is enabled."
    set_output false
    ;;
  *)
    echo "::warning::Could not check the dependency graph (HTTP $status); running dependency review anyway."
    set_output true
    ;;
esac
