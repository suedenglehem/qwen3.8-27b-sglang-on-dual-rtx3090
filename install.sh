#!/bin/bash
# install.sh — one-shot installer for the Qwen3.8-27B sglang serving stack.
#
# Builds, into the install root (default: this repo's directory):
#   1. sglang v0.5.20 (pinned commit) + the 33 local patches (repo/patches/)
#   2. a uv venv (python 3.12) with the exact pinned wheel set (torch cu130, etc.)
#   3. the model checkpoint (int4 or fp8) — from a predownloaded path or Hugging Face
#   4. config.env (host-specific paths; gitignored; written next to run_qwen38.sh)
# then verifies the install (venv import + DRY_RUN of the launcher).
#
# It does NOT start the server — that is `./run_qwen38.sh`.
#
# Usage:
#   ./install.sh [int4|fp8] [--model-dir PATH] [--install-dir PATH] [--dry-run]
#
#   int4            (default) AWQ int4 ship — hf-mattbucci/Qwen3.8-27B-AWQ
#   fp8             FP8 e4m3 official, whole repo, carries the MTP head — Qwen/Qwen3.8-27B-FP8
#   --model-dir P   use a predownloaded checkpoint at P (validated, used in place, never moved)
#   --install-dir P install root for venv/models/JIT (default: this script's directory).
#                   config.env is always written next to run_qwen38.sh (this dir) and points
#                   at the install root, so the two may differ (e.g. the 30 GB model on a
#                   bigger disk than the repo).
#   --dry-run       print what would run; change nothing
#
# Idempotent: re-running reuses the sglang tree (if already at the pinned commit + patched),
# the venv, and an already-downloaded model.
#
# See README.md (how) and PROGRESS.md (why: commit pin, venv recipe, traps, measured numbers).
set -euo pipefail

# ---------- constants ----------
SGLANG_TAG="v0.5.20"
SGLANG_COMMIT="94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
# Fingerprint of the fully-patched tree (33 patches on a pristine v0.5.20 clone).
EXPECTED_DIFF="38 files changed, 993 insertions(+), 113 deletions(-)"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/repo/patches"
RUN_SH="$SCRIPT_DIR/run_qwen38.sh"
CONFIG_ENV="$SCRIPT_DIR/config.env"

usage() {
    cat <<'USAGE'
Usage: ./install.sh [int4|fp8] [--model-dir PATH] [--install-dir PATH] [--dry-run]
  int4            (default) AWQ int4 ship — hf-mattbucci/Qwen3.8-27B-AWQ
  fp8             FP8 e4m3 official, whole repo, carries the MTP head — Qwen/Qwen3.8-27B-FP8
  --model-dir P   use a predownloaded checkpoint at P (validated, used in place, never moved)
  --install-dir P install root for venv/models/JIT (default: this script's directory)
  --dry-run       print what would run; change nothing
USAGE
}

# ---------- args ----------
VARIANT="int4"
MODEL_DIR_ARG=""
INSTALL_DIR=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        int4|fp8)      VARIANT="$1"; shift ;;
        --model-dir)   shift; MODEL_DIR_ARG="${1:?--model-dir needs a path}"; shift ;;
        --install-dir) shift; INSTALL_DIR="${1:?--install-dir needs a path}"; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"
VENV_DIR="$INSTALL_DIR/v0520"
SGLANG_DIR="$INSTALL_DIR/sglang-rebase-v0520"
MODELS_DIR="$INSTALL_DIR/models"
JIT_DIR="$INSTALL_DIR/.cache/sglang/jit"

case "$VARIANT" in
    int4) HF_REPO="hf-mattbucci/Qwen3.8-27B-AWQ"; MODEL_TARGET="$MODELS_DIR/hf-mattbucci/Qwen3.8-27B-AWQ" ;;
    fp8)  HF_REPO="Qwen/Qwen3.8-27B-FP8";        MODEL_TARGET="$MODELS_DIR/qwen3.8-27b-fp8" ;;
esac

# ---------- helpers ----------
die()  { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }
run()  {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  [dry-run] %s\n' "$*"
    else
        printf '  $ %s\n' "$*"
        "$@"
    fi
}

# ---------- 1. prerequisites ----------
step_prereqs() {
    step "Prerequisites"
    local c
    for c in git curl; do
        command -v "$c" >/dev/null 2>&1 || die "missing required tool: $c"
    done
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
        echo "  note: CUDA index order is host-specific and may not match nvidia-smi order."
        echo "        Before first launch, verify which indices are your two 24 GB cards"
        echo "        (README 'GPU selection'; PROGRESS.md §5)."
    else
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  [dry-run] nvidia-smi not found (would be fatal)"
        else
            die "nvidia-smi not found — install the NVIDIA driver first"
        fi
    fi
    if ! command -v uv >/dev/null 2>&1; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  [dry-run] curl -LsSf https://astral.sh/uv/install.sh | sh"
        else
            echo "  uv not found — installing ..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
            command -v uv >/dev/null 2>&1 || die "uv install did not put uv on PATH (check ~/.local/bin)"
        fi
    fi
    if command -v uv >/dev/null 2>&1; then echo "  uv: $(uv --version)"; fi
}

# ---------- 2. sglang tree ----------
apply_patches() {
    local p
    shopt -s nullglob
    local patches=("$PATCH_DIR"/*.patch)
    shopt -u nullglob
    [[ ${#patches[@]} -gt 0 ]] || die "no .patch files found in $PATCH_DIR"
    echo "  applying ${#patches[@]} patches (strict; any failure aborts, no partial state) ..."
    for p in "${patches[@]}"; do
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  [dry-run] git -C $SGLANG_DIR apply --check $(basename "$p") && git apply $(basename "$p")"
            continue
        fi
        git -C "$SGLANG_DIR" apply --check "$p" \
            || die "patch would not apply cleanly: $(basename "$p") — aborting"
        git -C "$SGLANG_DIR" apply "$p"
    done
    if [[ "$DRY_RUN" != "1" ]]; then
        local got
        got="$(git -C "$SGLANG_DIR" diff --shortstat | sed 's/^ *//')"
        [[ "$got" == "$EXPECTED_DIFF" ]] \
            || die "patched tree fingerprint '$got' != expected '$EXPECTED_DIFF'"
        echo "  patched tree verified: $got"
    fi
}

step_sglang() {
    step "sglang $SGLANG_TAG + patches"
    if [[ -d "$SGLANG_DIR/.git" ]]; then
        local head
        head="$(git -C "$SGLANG_DIR" rev-parse HEAD)"
        [[ "$head" == "$SGLANG_COMMIT" ]] \
            || die "$SGLANG_DIR is at $head, not the pinned $SGLANG_COMMIT — remove it or use --install-dir elsewhere"
        local cur
        cur="$(git -C "$SGLANG_DIR" diff --shortstat | sed 's/^ *//')"
        if [[ -z "$cur" ]]; then
            echo "  sglang tree present at pinned commit, unpatched — applying patches."
            apply_patches
        elif [[ "$cur" == "$EXPECTED_DIFF" ]]; then
            echo "  sglang tree already at pinned commit and fully patched — skipping."
        else
            die "sglang tree at pinned commit but in an unexpected state ('$cur'). Remove $SGLANG_DIR and re-run."
        fi
    else
        [[ -e "$SGLANG_DIR" ]] && die "$SGLANG_DIR exists but is not a git repo — remove it or use --install-dir elsewhere"
        run git clone --branch "$SGLANG_TAG" --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
        if [[ "$DRY_RUN" != "1" ]]; then
            local head
            head="$(git -C "$SGLANG_DIR" rev-parse HEAD)"
            [[ "$head" == "$SGLANG_COMMIT" ]] \
                || die "clone HEAD is $head, not the pinned $SGLANG_COMMIT (did the tag move?)"
        fi
        apply_patches
    fi
}

# ---------- 3. venv ----------
step_venv() {
    step "venv (uv, python 3.12)"
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        echo "  venv already present at $VENV_DIR — installing into it."
    else
        run uv venv --python 3.12 "$VENV_DIR"
    fi
    local py="$VENV_DIR/bin/python"
    run uv pip install --python "$py" torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url "$TORCH_INDEX"
    run env SGLANG_BUILD_RUST_EXTS=none uv pip install --python "$py" -e "$SGLANG_DIR/python"
    run uv pip install --python "$py" --no-deps transformers==5.12.1 gguf
    # hf_transfer: parallel-download accelerator for the model pull (optional).
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] uv pip install --python $py hf_transfer (optional)"
    else
        uv pip install --python "$py" hf_transfer 2>/dev/null \
            || echo "  note: hf_transfer not installed — model download will be slower (fine)."
    fi
    # Assert the editable install did not drift torch off the cu130 build.
    if [[ "$DRY_RUN" != "1" ]]; then
        local tv
        tv="$("$py" -c 'import torch; print(torch.__version__)')"
        if [[ "$tv" != *"+cu130"* ]]; then
            echo "  torch drifted to $tv (expected +cu130) — force-reinstalling the cu130 trio."
            run uv pip install --python "$py" --force-reinstall torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url "$TORCH_INDEX"
        else
            echo "  torch $tv (cu130) OK"
        fi
    fi
}

# ---------- 4. model ----------
model_present() {
    # fast presence check: config.json + >=1 safetensors (no full read)
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    local n
    n="$(find "$dir" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)"
    [[ "$n" -ge 1 ]]
}

validate_model() {
    local dir="$1"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] would validate model at $dir"
        return 0
    fi
    [[ -f "$dir/config.json" ]] || die "model dir has no config.json: $dir"
    local n
    n="$(find "$dir" -maxdepth 1 -name '*.safetensors' | wc -l)"
    [[ "$n" -ge 1 ]] || die "model dir has no safetensors: $dir"
    echo "  model: $n safetensors at $dir"
    if [[ "$VARIANT" == "fp8" && -f "$dir/crc32.txt" ]]; then
        echo "  crc32-verifying safetensors against $dir/crc32.txt (reads the whole checkpoint) ..."
        "$VENV_DIR/bin/python" - "$dir" <<'PY'
import sys, zlib, os
d = sys.argv[1]
man = {}
with open(os.path.join(d, "crc32.txt")) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        crc, name = line.split(None, 1)
        man[name] = crc.lower()
st_ok = st_bad = nw_ok = nw_bad = 0
bad = []
for name, crc in man.items():
    p = os.path.join(d, name)
    if not os.path.exists(p):
        bad.append((name, "MISSING"))
        continue
    actual = format(zlib.crc32(open(p, "rb").read()) & 0xffffffff, "08x")
    is_st = name.endswith(".safetensors")
    if actual == crc:
        st_ok += 1 if is_st else 0
        nw_ok += 0 if is_st else 1
    else:
        st_bad += 1 if is_st else 0
        nw_bad += 0 if is_st else 1
        bad.append((name, f"crc {crc} != {actual}"))
print(f"    safetensors: {st_ok} ok, {st_bad} bad")
print(f"    non-weight : {nw_ok} ok, {nw_bad} bad (a few stale-manifest files are expected to mismatch)")
for n, why in bad:
    print(f"    {n}: {why}")
sys.exit(1 if st_bad else 0)
PY
    elif [[ "$VARIANT" == "fp8" ]]; then
        echo "  note: no crc32.txt in $dir — skipping crc32 verification."
    fi
}

download_model() {
    local target="$1"
    local need_gb
    case "$VARIANT" in fp8) need_gb=35 ;; *) need_gb=25 ;; esac
    if [[ "$DRY_RUN" != "1" ]]; then
        local avail
        avail="$(df -BG "$INSTALL_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')"
        [[ "$avail" -ge "$need_gb" ]] \
            || die "need ~${need_gb} GB free on $INSTALL_DIR, found ${avail} GB"
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "  note: HF_TOKEN is not set — public repos download fine, but you may hit rate limits."
    fi
    run mkdir -p "$target"
    run env HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 "$VENV_DIR/bin/hf" download "$HF_REPO" --local-dir "$target"
}

step_model() {
    step "model ($VARIANT: $HF_REPO)"
    local model_dir
    if [[ -n "$MODEL_DIR_ARG" ]]; then
        model_dir="$(cd "$MODEL_DIR_ARG" 2>/dev/null && pwd)" || die "--model-dir not found: $MODEL_DIR_ARG"
        echo "  using predownloaded model at $model_dir (in place; not moved)"
    elif model_present "$MODEL_TARGET"; then
        model_dir="$MODEL_TARGET"
        echo "  model already present at $model_dir — reusing (skipping download)."
    elif [[ "$DRY_RUN" == "1" ]]; then
        model_dir="$MODEL_TARGET"
        echo "  [dry-run] would prompt for a predownloaded $VARIANT path, else download $HF_REPO -> $MODEL_TARGET"
    elif [[ ! -t 0 ]]; then
        die "stdin is not a TTY — pass --model-dir PATH, or run interactively to be prompted"
    else
        local reply
        read -r -p "  Predownloaded $VARIANT model at? (path, or Enter to download from Hugging Face $HF_REPO): " reply
        if [[ -n "$reply" ]]; then
            model_dir="$(cd "$reply" 2>/dev/null && pwd)" || die "path not found: $reply"
            echo "  using predownloaded model at $model_dir (in place; not moved)"
        else
            model_dir="$MODEL_TARGET"
            download_model "$model_dir"
        fi
    fi
    CHOSEN_MODEL_DIR="$model_dir"
    validate_model "$model_dir"
}

# ---------- 5. config.env ----------
step_config() {
    step "config.env"
    local int4_dir="$MODELS_DIR/hf-mattbucci/Qwen3.8-27B-AWQ"
    local fp8_dir="$MODELS_DIR/qwen3.8-27b-fp8"
    if [[ "$VARIANT" == "int4" ]]; then int4_dir="$CHOSEN_MODEL_DIR"; else fp8_dir="$CHOSEN_MODEL_DIR"; fi
    local body
    body="$(cat <<EOF
# config.env — host-specific overrides for run_qwen38.sh (gitignored).
# Written by install.sh on $(date +%Y-%m-%d) for the $VARIANT variant.
# Edit to taste; leave a variable unset to fall back to the portable default in run_qwen38.sh.

# Venv location (install root).
export VENV_DIR="$VENV_DIR"

# Model locations.
export MODELS_DIR="$MODELS_DIR"
export INT4_MODEL_DIR="$int4_dir"
export FP8_MODEL_DIR="$fp8_dir"

# Real path for the sgl-kernel JIT cache (NOT ~/.cache — see run_qwen38.sh).
export SGLANG_JIT_CACHE_DIR="$JIT_DIR"

# GPU selection — the two 24 GB cards. CUDA index order is host-specific; verify
# with torch.cuda.get_device_properties(i).uuid before changing (PROGRESS.md §5).
# export CUDA_VISIBLE_DEVICES="0,1"
EOF
)"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] would write $CONFIG_ENV:"
        echo "$body" | sed 's/^/    /'
    else
        if [[ -f "$CONFIG_ENV" ]]; then
            cp "$CONFIG_ENV" "$CONFIG_ENV.bak"
            echo "  existing config.env backed up to config.env.bak"
        fi
        echo "$body" > "$CONFIG_ENV"
        echo "  wrote $CONFIG_ENV"
    fi
}

# ---------- 6. verify ----------
step_verify() {
    step "verify"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [dry-run] would: import torch+sglang in the venv, list CUDA devices,"
        echo "            then DRY_RUN the launcher for the $VARIANT variant."
        return 0
    fi
    "$VENV_DIR/bin/python" - <<'PY'
import torch, sglang
print(f"  torch {torch.__version__}   sglang {getattr(sglang, '__version__', 'imported')}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/2**30:.1f} GB")
PY
    # run_qwen38.sh flags are -i4 / -fp8 (not -int4 / -fp8-variant) — map the
    # install variant onto the launcher's spelling.
    local run_flag
    case "$VARIANT" in
        int4) run_flag="-i4" ;;
        fp8)  run_flag="-fp8" ;;
    esac
    echo "  launcher argv (DRY_RUN, $VARIANT):"
    ( cd "$SCRIPT_DIR" && DRY_RUN=1 "$RUN_SH" "$run_flag" ) | sed 's/^/    /'
}

# ---------- main ----------
main() {
    echo "install.sh — Qwen3.8-27B sglang stack ($VARIANT)"
    echo "  install root : $INSTALL_DIR"
    echo "  repo dir     : $SCRIPT_DIR"
    echo "  dry-run      : $DRY_RUN"
    step_prereqs
    step_sglang
    step_venv
    step_model
    step_config
    step_verify
    local hint
    if [[ "$VARIANT" == "fp8" ]]; then hint="./run_qwen38.sh -fp8 -mtp"; else hint="./run_qwen38.sh"; fi
    echo
    echo "==> Done."
    echo "Next:"
    echo "  cd $SCRIPT_DIR"
    echo "  $hint"
    echo "  (first boot compiles the JIT kernel cache, ~12 min; subsequent boots ~55-70 s)"
    echo "  server listens on port 8091 (drop-in for the existing vLLM scripts)"
}
main
