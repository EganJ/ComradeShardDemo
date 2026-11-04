#!/bin/bash
# Usage: [VAR=value ...] ./run.sh       (main.jl's problem-size variables are listed in main.jl)
#   BACKEND  cpu (forced host CPU devices, default), gpu, tpu or trainium
#   GPU=1    same as BACKEND=gpu
#   NDEV     number of forced host CPU devices (default 4); BACKEND=cpu only
#   ACCEL_XLA_FLAGS  XLA_FLAGS for an accelerator backend (default empty; GPU_XLA_FLAGS also works)
#   JT       Julia threads (default 8)
#   TMO      timeout in seconds (default 3600)
HERE="$(cd "$(dirname "$0")" && pwd)"
export JULIA_LOAD_PATH="@:@v#.#:@stdlib"
export XLA_REACTANT_GPU_PREALLOCATE=false
BACKEND="${BACKEND:-$([ "${GPU:-0}" = "1" ] && echo gpu || echo cpu)}"
case "$BACKEND" in
    cpu)
        export XLA_FLAGS="--xla_force_host_platform_device_count=${NDEV:-4}"
        export CUDA_VISIBLE_DEVICES=""
        ;;
    gpu | tpu | trainium)
        export XLA_FLAGS="${ACCEL_XLA_FLAGS:-${GPU_XLA_FLAGS:-}}"
        ;;
    *)
        echo "BACKEND must be cpu, gpu, tpu or trainium, not '$BACKEND'" >&2
        exit 1
        ;;
esac
export DEMO_BACKEND="$BACKEND"
exec timeout --signal=TERM --kill-after=30 "${TMO:-3600}" \
    julia -t "${JT:-8}" --project="$HERE" "$HERE/main.jl" "$@"
