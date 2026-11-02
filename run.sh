#!/bin/bash
# Usage: [VAR=value ...] ./run.sh       (main.jl's problem-size variables are listed in main.jl)
#   NDEV   number of forced host CPU devices (default 4); ignored with GPU=1
#   GPU=1  use the visible GPUs instead of forced CPU devices
#   GPU_XLA_FLAGS  XLA_FLAGS for GPU=1 (default empty)
#   JT     Julia threads (default 8)
#   TMO    timeout in seconds (default 3600)
HERE="$(cd "$(dirname "$0")" && pwd)"
export JULIA_LOAD_PATH="@:@v#.#:@stdlib"
export XLA_REACTANT_GPU_PREALLOCATE=false
if [ "${GPU:-0}" = "1" ]; then
    export DEMO_BACKEND=gpu
    export XLA_FLAGS="${GPU_XLA_FLAGS:-}"
else
    export XLA_FLAGS="--xla_force_host_platform_device_count=${NDEV:-4}"
    export CUDA_VISIBLE_DEVICES=""
    export DEMO_BACKEND=cpu
fi
exec timeout --signal=TERM --kill-after=30 "${TMO:-3600}" \
    julia -t "${JT:-8}" --project="$HERE" "$HERE/main.jl" "$@"
