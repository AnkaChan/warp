#!/usr/bin/env bash
# Run validation in the isolated worktree, holding one GPU and logging clocks.
set -euo pipefail
cd /home/horde/Code/Graphics/warp-working-copies/ebvh-query-accel
source /home/horde/Code/AI-Docs/Envs/env-variables/ankac-dgxc-2gpu.env
source /home/horde/Code/AI-Docs/Envs/scripts/gpu-claim.sh ebvh-query-accel occupy
export WARP_CUDA_PATH="$PWD/_build/ebvh-cuda-12.6.3"
export WARP_CACHE_PATH="$PWD/_build/ebvh-kernel-cache-${EBVH_CACHE_TAG:-v1}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
ebvh_log_dir=/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation
ebvh_run_id=$(date -u +%Y-%m-%dT%H%M%S)-$$
ebvh_locked=0
export EBVH_CLOCK_MHZ="${EBVH_CLOCK_MHZ:-2490}"
ebvh_monitor_pid=
cleanup() {
    if [ -n "$ebvh_monitor_pid" ]; then
        kill "$ebvh_monitor_pid" 2>/dev/null || true
        wait "$ebvh_monitor_pid" 2>/dev/null || true
    fi
    if [ "$ebvh_locked" = 1 ]; then
        sudo -n nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -rgc || true
    fi
    gpu-release
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if sudo -n nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -lgc "$EBVH_CLOCK_MHZ"; then
    ebvh_locked=1
else
    echo '[EBVH] Clock lock unavailable; use interleaved repeats and recorded telemetry.'
fi
export EBVH_CLOCK_LOCKED="$ebvh_locked"
export EBVH_RUN_ID="$ebvh_run_id"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name,driver_version,clocks.max.sm --format=csv
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv -lms 500 > "$ebvh_log_dir/$ebvh_run_id-telemetry.csv" &
ebvh_monitor_pid=$!
"$@"
