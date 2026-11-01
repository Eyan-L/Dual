#!/usr/bin/env bash
set -euo pipefail
: "${SLACK_FINAL_ROOT:?}" "${SLACK_FINAL_PROFILE:?}" "${PBS_JOBID:?}"
module load singularity/4.3.1
unset PYTHONPATH PYTHONHOME CUDA_LAUNCH_BLOCKING
cd "$SLACK_FINAL_ROOT"
project=$(dirname "$SLACK_FINAL_ROOT")
case "$SLACK_FINAL_PROFILE" in
 cooperative_4gpu) stage=05_trace_serving; experiment=cooperative_trace_background ;;
 attention_2gpu) stage=04_flashinfer; experiment=nvlink_flashinfer_multiturn ;;
 controls/independent_100) stage=05_trace_serving; experiment=dual_trace_background ;;
 controls/native_matched) stage=05_trace_serving; experiment=dual_trace_capacity_control ;;
 *) exit 2 ;;
esac
result="$project/results/$stage/$experiment/${PBS_JOBID%%.*}"
mkdir -p "$result"
monitor_pid=
finish(){ rc=$?; if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi; printf '{"job_id":"%s","exit_code":%d}
' "$PBS_JOBID" "$rc" > "$result/pbs_exit.json"; }
trap finish EXIT
sha256sum -c SHA256SUMS > "$result/package_check.log"
cp "$SLACK_FINAL_PROFILE/config.json" "$result/submitted_config.json"
cp release.json "$result/package_release.json"
nvidia-smi -L > "$result/gpus.txt"
stdbuf -oL nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits --loop=1 > "$result/gpu_clocks.csv" &
monitor_pid=$!
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
cache="/tmp/slack_final_${PBS_JOBID}"
mkdir -p "$cache" "$project/runtime_cache/flashinfer"
singularity exec --cleanenv --nv --env HF_HUB_OFFLINE=1 --env OMP_NUM_THREADS=4 --env MAX_JOBS=4 \
 --env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
 --env "XDG_CACHE_HOME=$cache" --env "SLACK_FI_BUILD_ROOT=$project/runtime_cache/flashinfer" \
 --env 'LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/.singularity.d/libs' \
 --bind /scratch/users/ntu/n2602695:/scratch/users/ntu/n2602695 \
 /scratch/users/ntu/n2602695/sif/sglang_v0.5.21_cuda13.sif \
 /opt/sglang/bin/python3 -B -u "$SLACK_FINAL_ROOT/$SLACK_FINAL_PROFILE/code/run.py" \
 --config "$SLACK_FINAL_ROOT/$SLACK_FINAL_PROFILE/config.json" --out "$result/measurement" > "$result/driver.log" 2>&1
