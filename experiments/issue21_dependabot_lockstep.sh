#!/usr/bin/env bash
# Can Dependabot's grouped uv update move torch, torchvision and triton? (#21)
#
# torchvision X.Y.0 requires torch==A.B.0 exactly, and torch requires one
# triton release. Dependabot's uv ecosystem updates the members of a group
# one at a time (dependabot/dependabot-core#16390): for each it rewrites that
# member's pin in the .in file and runs
#   uv pip compile --build-isolation --output-file=<lock> --python-version=X
#     -P <dependency>==<version> <in-file>
# (uv/lib/dependabot/uv/file_updater/compile_file_updater.rb). This replays
# that on torch 2.13.0 / torchvision 0.28.0 / triton 3.7.1 moving to
# 2.14.1 / 0.29.1 / 3.8.0, with the trio
#   pinned   - each one pinned exactly in the .in, as before this change;
#   carried  - only torch pinned, torchvision named without a version and
#              triton left to torch, as scripts/unimedvl-requirements.in has it.
# A member whose compile fails is skipped, as Dependabot skips it.
#
#   bash experiments/issue21_dependabot_lockstep.sh [UV]
set -uo pipefail
uv=${1:-uv}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
compile() { # dir, extra uv arguments...
  (cd "$1" && shift && "$uv" pip compile --quiet --build-isolation \
    --output-file=req.txt --python-version=3.14 \
    --python-platform x86_64-manylinux_2_28 "$@" req.in)
}
trio() { grep -E '^(torch|torchvision|triton)==' "$1/req.txt" | tr '\n' ' '; echo; }
bump() { # dir, name, version: Dependabot's file update for one member
  sed -i -E "s/^$2==.*/$2==$3/" "$1/req.in"
  if out=$(compile "$1" -P "$2==$3" 2>&1); then
    echo "  $2 -> $3: lock now $(trio "$1")"
  else
    echo "  $2 -> $3: skipped, $(echo "$out" | grep -m1 -E 'Because|conclude' | sed 's/^ *//')"
    git -C "$1" checkout -q -- req.in req.txt
  fi
  git -C "$1" add -A && git -C "$1" commit -qm "$2" --allow-empty
}
run() { # name, .in content, order of members
  local dir="$work/$1"
  mkdir -p "$dir" && printf '%s\n' "$2" >"$dir/req.in"
  compile "$dir" || return 1
  git -C "$dir" init -q && git -C "$dir" add -A \
    && git -C "$dir" -c user.name=x -c user.email=x@x commit -qm start
  git -C "$dir" config user.name x && git -C "$dir" config user.email x@x
  echo "$1 (order: $3), start: $(trio "$dir")"
  for member in $3; do
    case $member in
      torch) bump "$dir" torch 2.14.1 ;;
      torchvision) grep -q '^torchvision==' "$dir/req.in" && bump "$dir" torchvision 0.29.1 ;;
      triton) grep -q '^triton==' "$dir/req.in" && bump "$dir" triton 3.8.0 ;;
    esac
  done
  echo "  result: $(trio "$dir")"
}
pinned=$'torch==2.13.0\ntorchvision==0.28.0\ntriton==3.7.1'
carried=$'torch==2.13.0\ntorchvision'
run pinned-a "$pinned" "torch torchvision triton"
run pinned-b "$pinned" "triton torchvision torch"
run carried-a "$carried" "torch torchvision triton"
run carried-b "$carried" "triton torchvision torch"
