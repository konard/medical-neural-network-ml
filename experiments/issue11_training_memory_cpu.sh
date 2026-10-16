#!/bin/bash
# Measure small full and LoRA training steps on a CPU (issue #9).
#
# Usage: experiments/issue11_training_memory_cpu.sh OUT_DIR [PYTHON]
#
# PYTHON needs torch, transformers and peft. The model files must already be in
# the Hugging Face cache (HF_HOME); the runs are offline. Each run is limited by
# `timeout` and by the RSS watchdog at 2.3 GB, below the 3.1 GB container limit,
# so the forced runs that do not fit are killed instead of exhausting the host.
# The results of 2026-10-16 are in
# docs/case-studies/issue-1/data/training-memory-cpu-20261016/.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
O=$(mkdir -p "$1" && cd "$1" && pwd)
P=${2:-python}
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
S=$ROOT/scripts/measure_training_memory.py
W="python3 -I $ROOT/experiments/issue11_rss_watchdog.py 2.3"
LM="--model HuggingFaceTB/SmolLM2-135M --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
VLM="--model HuggingFaceTB/SmolVLM-256M-Instruct --revision 7e3e67edbbed1bf9888184d9df282b700a323964"
common="--device cpu --autocast none --weights-dtype fp32 --steps 3"
cd "$O" || exit 1
run() {
  name=$1 limit=$2
  shift 2
  # shellcheck disable=SC2086 # $W and $P are word lists
  timeout "$limit" $W $P -I "$S" "$@" --out "$O/$name.json" > "$O/$name.log" 2>&1
  echo "$name rc=$?"
}
# shellcheck disable=SC2086 # $LM, $VLM and $common are word lists
{
run preflight-smollm2-135m-lora-r16-seq256 300 $LM --method lora --seq-len 256 $common
run smollm2-135m-qlora-seq256 300 $LM --method qlora --seq-len 256 $common --force
for rank in 8 16 64; do
  run smollm2-135m-lora-r$rank-seq256 900 $LM --method lora --lora-rank $rank --seq-len 256 $common --force
done
run smollm2-135m-lora-r16-seq1024 900 $LM --method lora --seq-len 1024 $common --force
run smolvlm-256m-img384-split-token-probe 300 $VLM --method lora --seq-len 64 --image-size 384 $common --force
run smolvlm-256m-lora-r16-img512-nosplit-seq256 1200 $VLM --method lora --seq-len 256 --image-size 512 \
  --processor-arg do_image_splitting=false $common --force
run smolvlm-256m-lora-r16-img512-split-seq1280 1500 $VLM --method lora --seq-len 1280 --image-size 512 $common --force
run smollm2-135m-full-seq256 900 $LM --method full --seq-len 256 $common --force
}
