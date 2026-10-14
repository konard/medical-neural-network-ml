#!/usr/bin/env bash
# Start llama-server once, report peak cgroup memory, then stop it (issue #1).
#
# Usage: bash experiments/issue_1_llama_server_memory_probe.sh LABEL [server args...]
#   LLAMA_DIR  folder with llama-server and its .so files
#   MODEL      language model GGUF
#   MMPROJ     optional vision projector GGUF
#   LOG_DIR    where LABEL.log is written (/tmp/exp)
#
# Used to find why llama-server was killed under a 3 GiB memory limit; results
# are in docs/case-studies/issue-1/measurements.md, section 3. The server runs
# at nice 10 and is stopped after it becomes healthy or after 120 s.
set -u
label=$1
shift
LLAMA_DIR="${LLAMA_DIR:-.cache/medbench/llama-b11609-f95cbb27dd6e}"
MODEL="${MODEL:-.cache/medbench/models/fa17ea2647ef/medgemma-4b-it-Q4_K_M.gguf}"
LOG_DIR="${LOG_DIR:-/tmp/exp}"
mkdir -p "$LOG_DIR"
log="$LOG_DIR/$label.log"

LD_LIBRARY_PATH="$LLAMA_DIR" nice -n 10 "$LLAMA_DIR/llama-server" -m "$MODEL" \
  ${MMPROJ:+--mmproj "$MMPROJ"} \
  --threads 4 --ctx-size 4096 --parallel 1 --host 127.0.0.1 --port 8090 "$@" \
  >"$log" 2>&1 &
pid=$!
peak=0
ok=no
rss=""
for i in $(seq 1 120); do
  cur=$(cat /sys/fs/cgroup/memory.current)
  ((cur > peak)) && peak=$cur
  rss=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null)
  if ! kill -0 "$pid" 2>/dev/null; then
    ok=died
    break
  fi
  if curl -sf http://127.0.0.1:8090/health >/dev/null; then
    ok=healthy
    break
  fi
  sleep 1
done
echo "$label: $ok after ${i}s, cgroup peak $((peak >> 20)) MiB, server RSS ${rss:-?} kB"
grep -iE "buffer size|repack|KV|mmproj|clip" "$log" | head -20
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
