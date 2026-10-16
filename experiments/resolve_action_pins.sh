#!/usr/bin/env bash
# Resolve each action tag to the commit it points at (issue #10).
# Usage: bash experiments/resolve_action_pins.sh owner/repo@tag ...
set -euo pipefail
for ref in "$@"; do
  repo=${ref%@*} tag=${ref#*@}
  obj=$(gh api "repos/$repo/git/ref/tags/$tag" --jq '.object.type + " " + .object.sha')
  type=${obj% *} sha=${obj#* }
  if [[ $type == tag ]]; then  # annotated tag: peel to the commit
    sha=$(gh api "repos/$repo/git/tags/$sha" --jq .object.sha)
  fi
  echo "$ref $sha"
done
