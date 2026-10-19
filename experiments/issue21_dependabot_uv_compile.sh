#!/usr/bin/env bash
# Recompile the UniMedVL lock the way Dependabot's uv ecosystem does (#21).
#
# dependabot-core uv/lib/dependabot/uv/file_updater/compile_file_updater.rb
# runs, in a copy of the update's directory (/scripts here):
#   uv pip compile --build-isolation --output-file=<lock> [--python-version=X]
#     -P <dependency>==<version> <in-file>
# keeping from the old header only --python-version (not --python-platform),
# and then puts the old header back. This prints the diff between that
# output and the committed lock; with the lock compiled from scripts/ only
# the header differs, compiled from the root every "# via -r" line would.
#
#   bash experiments/issue21_dependabot_uv_compile.sh [UV]
set -euo pipefail
uv=${1:-uv}
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$root/scripts/unimedvl-requirements.in" "$root/scripts/unimedvl-requirements.txt" "$work/"
version=$(grep -oE -- '--python-version[= ]+[0-9.]+' "$work/unimedvl-requirements.txt" | grep -oE '[0-9.]+$')
torch=$(grep -oE '^torch==[^ ]+' "$work/unimedvl-requirements.in")
(cd "$work" && "$uv" pip compile --quiet --build-isolation \
  --output-file=unimedvl-requirements.txt "--python-version=$version" \
  -P "$torch" unimedvl-requirements.in)
diff <(grep -v '^#' "$root/scripts/unimedvl-requirements.txt") \
  <(grep -v '^#' "$work/unimedvl-requirements.txt") && echo "Same lock as Dependabot would write."
