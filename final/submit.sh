#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
profile=${1:-cooperative_4gpu}
case "$profile" in cooperative_4gpu|attention_2gpu|controls/independent_100|controls/native_matched) ;; *) echo 'Unknown profile' >&2; exit 2 ;; esac
export SLACK_FINAL_ROOT="$root" SLACK_FINAL_PROFILE="$profile"
cd "$root/$profile"
qsub -v SLACK_FINAL_ROOT,SLACK_FINAL_PROFILE run.pbs
