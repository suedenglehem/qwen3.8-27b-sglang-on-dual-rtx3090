#!/bin/bash
# Qwen3.8-27B AWQ (mattbucci self-calibrated ship) on this host.
# Wraps repo/scripts/launch.sh's `qwen38` preset — all server flags come from there.
# See README.md in this directory for the 2-card vs 3-card run guide.
#
# Self-locating: every path resolves relative to this script's directory, so the repo can
# live anywhere. Host-specific values (model locations, JIT cache dir, chat template, GPU
# selection) come from config.env next to this script when present — install.sh writes it
# and it is gitignored. Without config.env the portable defaults below apply.
#
# Host adaptations vs the upstream rig (github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference):
#   - serving env: uv venv at <this dir>/v0520 (patched sglang v0.5.20) instead of conda
#   - CUDA_VISIBLE_DEVICES — the two 24 GB cards. CAUTION: CUDA index order is host-specific
#     and does not always match nvidia-smi order. On the original host, CUDA idx 0=nvidia-smi 0
#     (3090), CUDA 1=nvidia-smi 2 (3090), CUDA 2=nvidia-smi 1 (the display 3080 Ti, physically
#     between the two 3090s) — so "0,2" would silently grab the 3080 Ti. Do not "fix" the default
#     to match nvidia-smi indices; verify per host with torch.cuda.get_device_properties(i).uuid
#     (PROGRESS.md §5). Set it here or in config.env.
#   - MODELS_DIR / FP8_MODEL_DIR — model locations (see config.env)
#   - PORT=8091 — drop-in for the existing vLLM scripts on that port; upstream default is 23334
#   - CHAT_TEMPLATE_FILE — defaults to <this dir>/templates/qwen3.8.q6.sglang.jinja (the copy
#     of the user's own template that ships in this repo; a host may point it elsewhere via
#     config.env). It is NOT
#     stock: exactly one line is deliberately modified and that line is what makes Claude
#     Code accept the model. Pass it through VERBATIM — never edit, reformat, regenerate, or
#     "resync" it from the checkpoint's built-in template. The copy differs from the original
#     ONLY by accepting the full Anthropic effort ladder (adds max/high/minimal alongside the
#     original xhigh/medium/low; user-authorized 2026-10-13 — sglang's Anthropic adapter used
#     to rewrite xhigh→max, which made the original raise → HTTP 500 for Claude Code; that
#     rewrite is now patched out — see repo/patches/066-anthropic-effort-passthrough.patch).
#     The ORIGINAL qwen3.8.q6.jinja is
#     shared with llama-server and vLLM: never touch it, only re-copy if the user edits it.
#     Wired via OVERRIDE_ARGS (lands last in argv; launch.sh has no --chat-template CLI flag
#     of its own). CHAT_TEMPLATE_FILE= empty falls back to the checkpoint's embedded template.
#
#   - SGLANG_JIT_CACHE_DIR — defaults to <this dir>/.cache/sglang/jit, a REAL path for the
#     sgl-kernel JIT cache (its default is ~/.cache/sglang/jit, and on the original host
#     ~/.cache was a symlink to /dd2/vllm/.cache). sglang's kernels/jit/utils/compile/cache.py
#     compares resolved dependency
#     paths against an UNRESOLVED staging-dir string; under the symlinked spelling every build
#     records its .staging-<uuid>/cuda.cu into the deps key, so no published leaf ever hits and
#     each boot recompiles the marlin family (~12 min — most of the ~740 s warm boots). With
#     the real path leaves publish content-only keys and restarts load in seconds. The pre-fix
#     leaves already on disk carry poisoned keys; they are skipped cheaply, leave them or prune
#     at leisure. Report-worthy upstream: resolve build_dir in _to_entries (or cache_root).
#
# Usage:  ./run_qwen38.sh                 # start the server (foreground; Ctrl-C stops)
#         ./run_qwen38.sh -i4             # AWQ int4 ship — the DEFAULT (same as no flag)
#         ./run_qwen38.sh -fp8            # FP8 e4m3 checkpoint (see MODEL_VARIANT block below)
#         ./run_qwen38.sh -fp8 -mtp       # FP8 + MTP speculative decoding (checkpoint's own
#                                         # mtp.safetensors as NEXTN/EAGLE draft; ~2.5-3x decode).
#                                         # FP8-only: the AWQ ship carries no MTP weights.
#         ./run_qwen38.sh --all-gpus      # perf-test config: ALL three cards incl. the
#                                         # display 3080 Ti (PP=3). User-authorized exception
#                                         # to the 3090-only rule — never make this default.
#         DRY_RUN=1 ./run_qwen38.sh       # print the exact server argv without launching
#         ./run_qwen38.sh -fp8 -mtp --print-cmd  # same, as ONE shell-ready command line
#                                         # (pass last; composes with all flags above)
# Model variant, -mtp and --all-gpus compose freely (-fp8 -mtp --all-gpus = FP8+MTP on all three).
# Extra flags pass through to launch.sh, e.g. CTX/MEM overrides or --port 30001.
set -euo pipefail

# Self-locating root + host-specific overrides (install.sh writes config.env; gitignored).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    source "$SCRIPT_DIR/config.env"
fi

ALL_GPUS=0
MODEL_VARIANT=""
MTP=0
PRINT_CMD=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --all-gpus)  ALL_GPUS=1 ;;
        -i4|-fp8)    MODEL_VARIANT="$a" ;;   # last one wins if both are given
        -mtp)        MTP=1 ;;
        --print-cmd) PRINT_CMD=1 ;;
        *)           ARGS+=("$a") ;;
    esac
done

if [[ "$ALL_GPUS" == "1" ]]; then
    # Perf-test exception (user-authorized 2026-10-13): all three cards incl. the display
    # 3080 Ti. PP=3/TP=1, NOT TP=3: sglang's bootstrap rejects mixed-capacity TP groups
    # outright (_check_tp_memory_balance in srt/distributed/bootstrap.py) and the model's
    # 4 KV heads don't divide by 3 anyway. Verified booting (~70 s) and generating on all
    # three cards 2026-10-13. Expect LOWER throughput than the default: pipeline bubbles,
    # the slowest stage (the 3080 Ti, ~1.3 GB free after graph capture) gates every step,
    # and the KV pool shrinks to ~136K tokens vs ~697K on TP=2. Same port — one config at
    # a time; see README.md.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
    export OVERRIDE_ARGS="${OVERRIDE_ARGS:-} --tensor-parallel-size 1 --pipeline-parallel-size 3"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
fi

export MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"

# Model variant (default = -i4: the AWQ int4 ship, i.e. the launch.sh preset defaults).
case "$MODEL_VARIANT" in
    -fp8)
        # FP8 e4m3 dynamic checkpoint of the same qwen3_5 model — at FP8_MODEL_DIR
        # (config.env) or <this dir>/models/qwen3.8-27b-fp8 by default. Verified booting
        # + generating on TP=2 2026-10-13. NOTE: FP8 weights are ~2x the VRAM of INT4, so
        # the KV pool shrinks to ~259K tokens (max_total_num_tokens) vs ~697K for -i4 —
        # still covers the 262144 context claim only just.
        export MODEL="${FP8_MODEL_DIR:-$SCRIPT_DIR/models/qwen3.8-27b-fp8}"
        export QUANT=fp8
        ;;
    -i4)
        # Explicit spelling of the default (protects against preset-default drift).
        # INT4_MODEL_DIR (config.env) overrides the default <MODELS_DIR>/hf-mattbucci/...
        export MODEL="${INT4_MODEL_DIR:-$MODELS_DIR/hf-mattbucci/Qwen3.8-27B-AWQ}"
        export QUANT=awq_marlin
        ;;
esac

# MTP speculative decoding (opt-in): the checkpoint's own mtp.safetensors as a NEXTN/EAGLE
# draft (NEXTN is sglang's reserved alias for EAGLE; Qwen3_5 is not in the built-in-MTP
# auto-default arch list, so the draft path must be passed explicitly — same dir as target).
# FP8-only: the AWQ ship's index has zero mtp tensors. --disable-overlap-schedule +
# extra_buffer are the spec-family boot requirements on this DeltaNet hybrid (launch.sh
# DSPARK block comment). The two memory knobs are load-bearing, not tuning: the draft model
# costs 2.70 GB/GPU of weights and the mamba spec intermediate buffer ~0.58 GB/GPU, so at the
# preset defaults (MEM=0.85, K=8) the KV pool collapses to ~42K tokens; MEM=0.93 + K=6
# restores it to ~165K (measured 165,532). Verified 2026-10-13 on TP=2: accept length
# ~2.6/4 on code+reasoning, ~3.8/4 on easy text (~92% accept rate), up to ~92 tok/s decode;
# a 32K-token prefill (four 8K chunks) survives the reduced headroom. See README.md.
if [[ "$MTP" == "1" ]]; then
    if [[ "$MODEL_VARIANT" != "-fp8" ]]; then
        echo "ERROR: -mtp needs the FP8 checkpoint (-fp8); the AWQ ship has no MTP weights" >&2
        exit 1
    fi
    export OVERRIDE_ARGS="${OVERRIDE_ARGS:-} --speculative-algorithm NEXTN --speculative-draft-model-path $MODEL --disable-overlap-schedule --mamba-radix-cache-strategy extra_buffer --mem-fraction-static 0.93 --max-mamba-cache-size 6"
fi

export PORT="${PORT:-8091}"
# Real path, not ~/.cache — see the SGLANG_JIT_CACHE_DIR note in this header.
export SGLANG_JIT_CACHE_DIR="${SGLANG_JIT_CACHE_DIR:-$SCRIPT_DIR/.cache/sglang/jit}"

# Server-wide default sampling (repo/patches/067-server-default-sampling-params.patch):
# applies only where a request OMITS the value — explicit client params always win.
# Covers /v1/chat/completions and /v1/messages; presence/frequency entries are no-ops
# kept for explicitness (protocol default is already 0.0). max_new_tokens lands only when
# neither max_completion_tokens nor max_tokens is sent (Claude Code sends its own).
DEFAULT_SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0,"max_new_tokens":16384}'
export SGLANG_DEFAULT_SAMPLING_PARAMS="${SGLANG_DEFAULT_SAMPLING_PARAMS-$DEFAULT_SAMPLING}"

CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE-$SCRIPT_DIR/templates/qwen3.8.q6.sglang.jinja}"
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
    if [[ ! -r "$CHAT_TEMPLATE_FILE" ]]; then
        echo "ERROR: chat template not readable: $CHAT_TEMPLATE_FILE" >&2
        exit 1
    fi
    export OVERRIDE_ARGS="${OVERRIDE_ARGS:-} --chat-template $CHAT_TEMPLATE_FILE"
fi

# VENV_DIR (config.env) overrides the venv location so install.sh can build it on
# a different (bigger) disk than the repo; defaults to <this dir>/v0520.
source "${VENV_DIR:-$SCRIPT_DIR/v0520}/bin/activate"

if [[ "$PRINT_CMD" == "1" ]]; then
    # Print the exact server command as one shell-ready line and exit. launch.sh's DRY_RUN
    # prints argv one per line after its banner; drop everything up to and including the
    # closing ==== border, rejoin with %q quoting so the output is copy-paste runnable.
    mapfile -t _argv < <(DRY_RUN=1 "$SCRIPT_DIR/repo/scripts/launch.sh" qwen38 "${ARGS[@]}" \
        | awk '/^=+$/ {b = NR; next} {a[NR] = $0} END {for (i = b + 1; i <= NR; i++) if (i in a) print a[i]}')
    [[ ${#_argv[@]} -gt 0 ]] || { echo "ERROR: --print-cmd got no argv from launch.sh" >&2; exit 1; }
    printf '%q ' "${_argv[@]}"
    echo
    exit 0
fi

exec "$SCRIPT_DIR/repo/scripts/launch.sh" qwen38 "${ARGS[@]}"
