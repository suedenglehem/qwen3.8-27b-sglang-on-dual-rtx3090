# PROGRESS — Qwen3.8-27B on sglang v0.5.20, dual RTX 3090

This is the durable knowledge base for this project: what the stack is, how it was built,
the traps that cost hours, and the measured numbers. Read `README.md` for how to run it;
read this for *why* it is shaped the way it is. Provenance: built on top of mattbucci's 2×3090 rig
(`url.txt` → github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference), pinned at commit
`9e533ae` + a 3-file delta (`upstream-delta/`), adapted to this host (no conda, uv venv) and
to serving Qwen3.8-27B specifically.

---

## 1. What this is

Serves **Qwen3.8-27B** (a `qwen3_5` / `Qwen3_5ForConditionalGeneration` model, 64 layers,
16 full-attention + 48 GDN/DeltaNet hybrid, 24 attn / 4 KV heads, head_dim 256, hidden 5120,
max_ctx 262144) on **two RTX 3090 (24 GB each)** with a patched **sglang v0.5.20** tree.

Two quantizations of the same model are supported, selected at launch:

| Variant | Checkpoint | Quant | Notes |
|---|---|---|---|
| `-i4` (default) | `hf-mattbucci/Qwen3.8-27B-AWQ` | awq_marlin (INT4) | mattbucci self-calibrated ship; **no MTP weights** |
| `-fp8` | `Qwen/Qwen3.8-27B-FP8` | fp8 (e4m3, dynamic act) | official Qwen release; **full VLM** (vision+video); carries `mtp.safetensors` |

`-mtp` (speculative decoding) is **FP8-only** — it uses the FP8 checkpoint's own
`mtp.safetensors` as a NEXTN/EAGLE draft.

One entry point: **`./run_qwen38.sh`**. It wraps `repo/scripts/launch.sh`'s `qwen38` preset.

---

## 2. Repo layout (what ships vs what install builds)

```
README.md            # how to run (deploy guide)
PROGRESS.md          # this file
install.sh           # installs int4 | fp8 (clones rig + sglang, patches, builds venv, fetches model)
run_qwen38.sh        # self-locating launcher; reads config.env for host-specific paths
config.env           # host-specific paths (gitignored; install.sh writes it)
url.txt              # provenance: mattbucci rig URL
templates/qwen3.8.q6.sglang.jinja   # the chat template we serve (verbatim — see §6)
upstream-delta/      # the 3-file delta on the upstream rig (patches 066/067 + common.sh.patch)
```

**Built / downloaded at install time (git-excluded):**

```
repo/                  # the serving rig: upstream mattbucci @ 9e533ae + our delta (cloned + delta-applied by install.sh)
sglang-rebase-v0520/   # pristine sglang v0.5.20 clone + 33 patches applied (206 MB)
v0520/                 # uv venv, python 3.12.3 (8.6 GB)
models/                # downloaded model weights (int4 ~20 GB / fp8 ~31 GB)
.cache/sglang/jit/     # JIT kernel cache (path-keyed → machine-specific, never shipped)
```

The only *shipped* content is the small `upstream-delta/` (3 patches + a README) plus the
top-level scripts and docs. `install.sh` clones the upstream rig at a pinned commit and
applies the delta to rebuild `repo/`, then clones + patches sglang, builds the venv, and
fetches the model — so a fresh clone + `./install.sh fp8` reproduces the whole stack.

---

## 3. Stack details (the exact, reproducible recipe)

### 3.1 sglang source — pinned, not "latest"

- Repo: `https://github.com/sgl-project/sglang.git`
- Tag: **`v0.5.20`** = commit **`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`** (verified
  `git tag --points-at HEAD` = `v0.5.20`).
- The live tree = **pristine clone @ v0.5.20 + the 33 patches in `repo/patches/*.patch`**
  applied in glob order via `git apply`. All but **066/067** come straight from the upstream
  rig; **066/067 come from our delta** (`upstream-delta/`, applied by `install.sh` `step_rig`).
  Result: **38 files changed, 993 insertions(+), 113 deletions(-), 0 untracked**. This was
  verified byte-identical to the live tree by mattbucci's 3-gate pristine replay (see
  `repo/patches/README.md`).
- 33 `.patch` files = 31 logical units (a couple of patches split a single logical change).
  `install.sh` applies them **strictly** on a fresh clone: any `git apply --check` failure
  aborts (a silent skip on a pristine clone can only mean a broken chain).

### 3.2 venv — uv, not conda

This host has **no conda**. `repo/scripts/common.sh` was adapted: when a venv is active
(`VIRTUAL_ENV` set) it sets `SGLANG_NO_CONDA=1` and `activate_conda` becomes a no-op, so
`launch.sh` runs unmodified. The venv is a **uv venv at `v0520/`, python 3.12.3**.

Exact pins (all verified in the live venv):

| Package | Version | Source |
|---|---|---|
| torch | **2.13.0+cu130** | `https://download.pytorch.org/whl/cu130` |
| torchvision | 0.28.0+cu130 | same cu130 index |
| torchaudio | 2.11.0+cu130 | same cu130 index |
| triton | 3.7.1 | (transitive of torch) |
| transformers | **5.12.1** | pinned `--no-deps` — do NOT let it drift |
| gguf | 0.19.0 | `pip install gguf` |
| flashinfer-python | 0.6.18 | [cu13] |
| sglang-kernel | 0.4.7 | |
| xgrammar | 0.2.1 | |
| huggingface_hub | 1.33.0 | ships the `hf` CLI in the venv bin |
| sglang | 0.5.20 | **editable** install from `sglang-rebase-v0520/python` |

Build commands (what `install.sh` runs):

```bash
uv venv --python 3.12 v0520
source v0520/bin/activate
uv pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130
SGLANG_BUILD_RUST_EXTS=none uv pip install -e ./sglang-rebase-v0520/python
uv pip install --no-deps transformers==5.12.1 gguf
```

- **`SGLANG_BUILD_RUST_EXTS=none`** is load-bearing: v0.5.20's setup.py otherwise builds a
  mandatory Rust gRPC extension that needs `protoc`/`cargo` (absent here). We serve over
  HTTP, so dropping the ext is correct. (This replaces mattbucci's retired patch 037.)
- **transformers is pinned exactly at 5.12.1** with `--no-deps`. Newer transformers
  re-routes tokenizer backends (e.g. 5.12 sends Mistral-family checkpoints to
  `MistralCommonBackend`, which patch 057 fixes). Do not bump it without re-probing quality.
- After the editable install, `install.sh` asserts `torch.__version__` still contains
  `+cu130` (force-reinstalls the trio from the cu130 index if the resolver drifted it).

### 3.3 The qwen38 preset (`repo/scripts/launch.sh`)

The `qwen38` preset (around `launch.sh:473`) sets, for both variants:

```
CTX=262144   MEM=0.85   MAX_RUNNING=1   CHUNKED=8192   DECODE_STEPS=32
--max-mamba-cache-size 8
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
--cuda-graph-max-bs-decode 1 --cuda-graph-backend-prefill disabled
```

- **`MAX_RUNNING=1` is load-bearing.** It is a single-user 256K-context rig. Raising it to 8
  collapses the KV pool from ~697K to ~32K tokens (measured). Do not "fix" it.
- `--tool-call-parser qwen3_coder` is per-preset and load-bearing (mattbucci rule): without
  the right parser the model's tool-call XML is served as plain content and agent harnesses
  drop it.

---

## 4. Model sources (Hugging Face)

### 4.1 FP8 — `Qwen/Qwen3.8-27B-FP8` (official release)

- **Single-artifact repo — fetch the WHOLE repo, no `files:` filter.** The layout is native
  to the repo (not a repack): 66 `layers-0..63.safetensors` + `outside.safetensors`
  (embed/lm_head + 333 `visual.*` tensors) + `mtp.safetensors` + tokenizer + preprocessor
  configs.
- Size: **30,866,866,928 bytes = 30.87 GB decimal = 28.75 GiB** (weights only).
- Downloaded at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- **Integrity:** the repo ships its own `crc32.txt` manifest. All 66 safetensors verify
  against it (`install.sh` re-verifies after download via `zlib.crc32`). Three *non-weight*
  files mismatch the stale manifest — known, not corruption.
- Quant config: `quant_method: fp8`, `e4m3`, `activation_scheme: dynamic`,
  `weight_block_size: [128,128]`.
- **FP8 is a FULL VLM.** `modules_to_not_convert` excludes every `visual.blocks.*`, so the
  vision tower stays higher-precision while the LLM is FP8. `preprocessor_config.json` +
  `video_preprocessor_config.json` ship in-repo → vision and video serve with **no extra
  download**. (Never exercised on this host — text-only so far — but the capability is in
  the checkpoint.)
- Arch: `Qwen3_5ForConditionalGeneration`.

### 4.2 INT4 — `hf-mattbucci/Qwen3.8-27B-AWQ`

- mattbucci's self-calibrated AWQ ship (his own calibration, not a community AWQ).
- This host's copy: `/dd2/lmstudio/Models/sglang/hf-mattbucci/Qwen3.8-27B-AWQ`.
- **Zero `mtp` tensors** in the weight index → `-i4 -mtp` fails fast with a clear error
  rather than dying mid-boot.

---

## 5. GPU mapping (host-specific — re-verify on every new host)

On **this** host, CUDA enumerates differently from nvidia-smi (verified by UUID):

| CUDA index | nvidia-smi index | Card |
|---|---|---|
| 0 | 0 | RTX 3090 |
| 1 | 2 | RTX 3090 |
| 2 | 1 | RTX 3080 Ti (display, physically between the two 3090s) |

So `CUDA_VISIBLE_DEVICES=0,1` = the two 3090s; `0,1,2` = all cards. Writing `0,2` would
silently grab a 3090 **and the display card**. **Do not "fix" this to match nvidia-smi
indices** — it is correct as-is for this host.

**On a new host the ordering is unknown.** Before first launch, verify which CUDA indices
are the two 24 GB cards:

```bash
python -c "import torch; [print(i, torch.cuda.get_device_properties(i).uuid,
    torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

then set `CUDA_VISIBLE_DEVICES` (or the commented line in `config.env`) to the two 24 GB
cards. `run_qwen38.sh` defaults to `0,1` and honors an existing `CUDA_VISIBLE_DEVICES`.

---

## 6. Chat template (load-bearing — pass verbatim)

We serve with `templates/qwen3.8.q6.sglang.jinja`
(md5 **`dc62ec63f1d888c7456614d17b92cdb3`**). It is a **copy** of the user's own template
(the one llama-server runs this checkpoint with), passed **verbatim**.

- It is **not stock**: exactly one block is deliberately modified, and that block is what
  makes Claude Code accept the model. The diff vs the shared original is *only* the
  reasoning-effort ladder: it accepts `xhigh,max,high,medium,low,minimal` (the original
  accepts `xhigh,medium,low`), with branches grouped `in ('xhigh','max','high')` and
  `in ('low','minimal')` and the effort name interpolated.
- History: sglang's Anthropic adapter used to rewrite `xhigh→max`, which made the original
  template raise → HTTP 500 for Claude Code. That rewrite is now patched out
  (`repo/patches/066-anthropic-effort-passthrough.patch`), and the template was widened to
  the full ladder (user-authorized 2026-10-13).
- **The ORIGINAL `qwen3.8.q6.jinja` is shared with llama-server and vLLM — never touch it.**
  Only re-copy into `templates/` if the user edits the original.
- Wired via `OVERRIDE_ARGS` (lands last in argv; `launch.sh` has no `--chat-template` flag
  of its own). `CHAT_TEMPLATE_FILE=` empty falls back to the checkpoint's embedded template.

---

## 7. FP8 + MTP memory model (why the `-mtp` flags are what they are)

`-mtp` runs the checkpoint's own `mtp.safetensors` (one full-attention layer, FP8 in the
checkpoint — sglang inherits the quant config) as a NEXTN/EAGLE draft: 3 steps, topk 1,
4 draft tokens per step. The flag appends to `OVERRIDE_ARGS`:

```
--speculative-algorithm NEXTN --speculative-draft-model-path <fp8 dir> \
--disable-overlap-schedule --mamba-radix-cache-strategy extra_buffer \
--mem-fraction-static 0.93 --max-mamba-cache-size 6
```

- `NEXTN` is sglang's reserved alias for EAGLE. Qwen3_5 is **not** in the built-in-MTP
  auto-default arch list, so the draft path must be passed explicitly (same dir as target).
- `--disable-overlap-schedule` + `extra_buffer` are the spec-family boot requirements on
  this DeltaNet hybrid.

**The last two flags are load-bearing, not tuning.** The draft model costs **2.70 GB/GPU**
of weights and the mamba spec intermediate buffer another **~0.58 GB/GPU**. At the preset
defaults (`MEM=0.85`, `--max-mamba-cache-size 8`) that collapses the KV pool to **~42K
tokens**. Raising `--mem-fraction-static` to **0.93** and dropping the mamba cache from 8 to
**6** slots (no concurrency loss: with `MAX_RUNNING=1` any K≥4 suffices) restores the pool
to **165,532 tokens** (measured `max_total_num_tokens`).

- **K (mamba cache size) is a weak lever; MEM (mem-fraction) is the strong lever.**
- **Raising CTX alone does NOT help** — context length is only an upper bound on pool size,
  which is set by free VRAM.
- A 32K-token prefill (four 8K chunks) survives the reduced headroom. Requests past ~165K
  total tokens need plain `-fp8` (~259K) or `--all-gpus`.

**Measured 2026-10-13, TP=2:** mean accepted length **~2.6/4** on code+reasoning text,
**~3.8/4 (~92% accept rate)** on easy text — roughly a **2.5–3× decode speedup** over plain
FP8 (~85 tok/s end-to-end, up to ~92 tok/s decode).

---

## 8. Measured KV pools (`max_total_num_tokens`, TP=2 unless noted)

| Config | Pool |
|---|---|
| `-i4` (AWQ) | **~697,368** tokens |
| `-fp8` | **~259K** tokens |
| `-fp8 -mtp` (MEM=0.93, K=6) | **165,532** tokens |
| `--all-gpus` (PP=3, incl. 3080 Ti) | **~136K** tokens |

FP8 weights cost ~2× the VRAM of INT4, so its pool only just covers the 262144 context
claim. Deep requests past the pool fail with an out-of-memory-on-token error, not a crash.

---

## 9. JIT cache — the real-path fix (cost ~12 min per boot when broken)

sglang's `srt` kernels JIT-compile on first use per machine. The cache dir default is
`~/.cache/sglang/jit`. **On this host `~/.cache` is a symlink to `/dd2/vllm/.cache`**, and
`run_qwen38.sh` sets `SGLANG_JIT_CACHE_DIR` to the **real** path
(`/dd2/vllm/.cache/sglang/jit`, ~41 MB).

The bug: `sglang/.../kernels/jit/utils/compile/cache.py` compares resolved dependency paths
against an **unresolved** staging-dir string. Under the symlinked spelling, every build
records its `.staging-<uuid>/cuda.cu` into the deps key, so no published leaf ever hits and
each boot recompiles the marlin family (**~12 min** — most of the ~740 s "warm" boots). With
the real path, leaves publish content-only keys and restarts load in seconds. Pre-fix leaves
on disk carry poisoned keys; they are skipped cheaply. **Report-worthy upstream:** resolve
`build_dir` in `_to_entries` (or `cache_root`).

**The JIT cache is path-keyed → machine-specific. It never transfers between hosts.**
`install.sh` gives each host its own `<install>/.cache/sglang/jit` (a real path by
construction, which also sidesteps the symlink bug). First-ever boot on a fresh cache is
slow (~12 min); subsequent boots ~55–70 s.

---

## 10. Conventions (do not drift)

- **Port 8091** is the default — it is a drop-in for the existing vLLM scripts on that port
  (upstream default is 23334). Test servers run on **:8092**.
- **`--all-gpus` is a user-authorized perf-test exception only — never the default.** It
  pulls in the display 3080 Ti (PP=3). The default is the two 3090s only.
- **Server-wide default sampling** (patch `067-server-default-sampling-params.patch`, wired
  via `SGLANG_DEFAULT_SAMPLING_PARAMS` in `run_qwen38.sh`):
  `{"temperature":0.6,"top_p":0.95,"top_k":20,"presence_penalty":0.0,"frequency_penalty":0.0,
  "repetition_penalty":1.0,"max_new_tokens":16384}`.
  Applies **only where a request omits the value** — explicit client params always win
  (Claude Code sends its own `max_tokens`). Covers `/v1/chat/completions` and `/v1/messages`;
  not `/v1/completions` or `/v1/responses`. Override per launch with your own env var.
- **One config at a time** — same port; stop one before starting the other.

---

## 11. Why PP=3 and not TP=3 for the 3-card config

TP=3 is impossible on this hardware (verified by boot):

- sglang's bootstrap hard-rejects mixed-capacity tensor-parallel groups
  (`_check_tp_memory_balance` in `srt/distributed/bootstrap.py`) — a 12 GB card next to two
  24 GB cards fails the memory-balance check before anything else.
- The model has **4 KV heads**, which don't divide by 3 (TP=3 would need head splitting).

PP=3 splits the 64 layers across the three cards and boots cleanly (~70 s). Expect **lower
throughput than TP=2**: pipeline bubbles, the slowest stage (the 3080 Ti, only ~1.3 GB free
after CUDA graph capture) gates every decode step, and the KV pool is a quarter of the
2-card size. `--all-gpus` exists to measure whether it wins anyway.

---

## 12. Ops notes

- **Foreground:** `./run_qwen38.sh` (Ctrl-C stops). **Detached:**
  `setsid bash -c './run_qwen38.sh' > server.log 2>&1 & echo $!` — note the PID.
- **Stop:** `kill <PID>` (sglang tears down its children), then `nvidia-smi` — all cards back
  to ~0 MiB = clean. Teardown takes >4 s; use a bounded wait loop.
- **`setsid` forks**, so the recorded PID can die while the server lives — to *find* a
  running server use `pgrep -af 'sglang.launch_serve[r]'` (the bracket trick avoids
  pgrep matching its own command line).
- **`DRY_RUN=1 ./run_qwen38.sh [flags]`** prints the exact server argv without launching
  (one arg per line, after a banner). **`./run_qwen38.sh [flags] --print-cmd`** (pass last)
  prints it as one shell-ready, `%q`-quoted command line.
- Extra flags pass through to `launch.sh` (e.g. `--port 30001`, CTX/MEM overrides).

---

## 13. Mycelium memory references

Project `sglang-qwen38`: checkpoints **#153–#161** cover the FP8+MTP OOM fix, the JIT
cache real-path fix, `--print-cmd`, and this deploy packaging.

---

## 14. Session log

- **2026-10-13** — FP8+MTP OOM fixed (MEM 0.85→0.93, K 8→6 → 165,532 tokens). Added
  `--print-cmd` to `run_qwen38.sh`. Verified FP8 = full VLM (333 visual tensors). Grabbed
  the chat template into `templates/` (md5-verified).
- **2026-10-14** — Deploy packaging: this file + rewritten `README.md` + self-locating
  `run_qwen38.sh`/`config.env` + `install.sh` (int4|fp8). Cleaned test artifacts + stale
  `v312/` venv (8.7 GB, unreferenced).
  - **`install.sh` verified (all on this host):** `fp8 --dry-run` + `int4 --dry-run` both
    sane/complete. Fresh-clone replay in `/tmp` (clone @ pin + 33 strict patches) → shortstat
    fingerprint exact (`38 files / 993+ / 113-`) and working tree **byte-identical** to the
    live tree (only 2 diffs = gitignored editable-install artifacts the installer regenerates).
    Model validators pass (fp8 crc32 66/66 clean, 3 known-stale non-weight; int4 dir =
    config.json + single 18.7 GB `model.safetensors`). `run_qwen38.sh` argv gate: all 5 flag
    combos byte-identical to baseline after the `VENV_DIR` edit. Live server (PID 10431,
    :8091) untouched throughout.
  - **Deploy git repo initialized (no push yet):** `repo/.git` stripped; `.gitignore` written
    (fixed an inline-comment bug — gitignore `#` only comments at line start — that had staged
    `sglang-rebase-v0520` as a gitlink; `v0520/` self-ignores via its own uv-written
    `.gitignore`); initial commit `45169da` (914 files, 9.8 MB, nothing >100 MB, ends with the
    `Co-Authored-By` line); remote `origin` added.
  - **Next unfinished task:** user preps the i7 host (2× 24 GB GPUs, driver, internet,
    ~35 GB free disk for fp8 / ~25 GB int4). Test-install there: copy the repo (or clone once
    pushed) → `./install.sh fp8` (or `int4`) → `./run_qwen38.sh -fp8 -mtp` (or `./run_qwen38.sh`
    for int4). **Then** push to `git@github.com:suedenglehem/qwen3.8-27b-sglang-on-dual-rtx3090.git`.
- **2026-10-14 (i7 test install)** — Reproduced the stack on the user's i7 host
  (`ssh andrei@i7` = `i714700k`) into `/dd2/sglang`, per the user's ask: fit **int4** onto the
  i7's **2× ~16 GB** cards and launch at **140k context** ("it's a try, i won't cry if it
  doesn't work" — the user runs llama-server q4 of this model at 256k, so 140k is a
  conservative start for the heavier sglang stack).
  - **i7 hardware:** 2× ~16 GB **Ada Lovelace (sm_89)** — `cuda:0` RTX 4080 SUPER (16376 MiB),
    `cuda:1` RTX 4060 Ti (16380 MiB). 4 MiB gap → TP=2 memory-balance check passes. No third
    card, so `CUDA_VISIBLE_DEVICES=0,1` is unambiguous (no display-card trap like the 3090
    host). 28 cores, 62 GB RAM, /dd2 80 GB free. No system torch; `install.sh` auto-installed
    uv 0.13.0 to `~/.local/bin`.
  - **`install.sh int4` ran to completion** (3rd attempt). Model was **pre-existing** at
    `/mnt/models_sas_ssd/lmstudio/Models/sglang/hf-mattbucci/Qwen3.8-27B-AWQ` (passed via
    `--model-dir`, used in place, never moved). sglang clone @ pin + 33 strict patches
    (fingerprint `38 files / 993+ / 113-` verified), venv built (torch 2.13.0+cu130, sglang
    0.5.20), `config.env` written, verify passed (correct 2 CUDA devices + valid DRY_RUN argv).
  - **Two `install.sh` bugs fixed** — both missed by the earlier `--dry-run`-only testing
    (dry-runs never passed `--model-dir`, and the real verify step was never exercised):
    1. **Arg parsing** — `for a in "$@"` + `shift` is broken: the `for` captures the full list
       up front, so `shift` doesn't stop the loop from later visiting a `--model-dir` path as a
       stray positional → `ERROR: unknown argument: /mnt/...`. Rewrote as the
       `while [[ $# -gt 0 ]]; do case "$1" in ... shift ;; esac; done` idiom.
    2. **`step_verify` variant name** — ran `run_qwen38.sh "-$VARIANT"` → `-int4`, but the
       launcher only accepts `-i4`/`-fp8` → `Unknown option: -int4`. Now maps `int4→-i4`,
       `fp8→-fp8`.
  - **Launch (the "try"):** `./run_qwen38.sh -i4 --context-length 140000`. The qwen38 preset
    hardcodes `CTX=262144` (not env-overridable), so the `--context-length` CLI flag — applied
    *after* the preset in `launch.sh` — is the override (confirmed in the DRY_RUN argv). Memory
    budget for 140k on 2× 16 GB: int4 weights ~9.4 GB/card + KV@140k ~2.2 GB/card (16
    full-attn layers × 4 KV heads × 256 head_dim × 2 × 1 B fp8 = 32 KB/token) ≈ 11.6 GB/card
    vs the `MEM=0.85` budget of 13.6 GB/card → ~2 GB headroom for CUDA graphs / mamba state /
    activations. **It fit and booted.** Weights 8.80 GB/card, KV pool **215,536 tokens**
    (K+V 3.28 GB/card, fp8), mamba 0.63 GB/card → ~2.4 GB free/card after pools; decode CUDA
    graph captured, server healthy on :8091. Generation verified coherent ("capital of France"
    → "Paris", `finish_reason: stop`). The 140k cap sits well under the 215k pool, so there is
    headroom to raise `--context-length` toward ~215k if wanted.
  - **Next unfinished task:** commit the 2 `install.sh` fixes + this entry, then **push** to
    `git@github.com:suedenglehem/qwen3.8-27b-sglang-on-dual-rtx3090.git`.
- **2026-10-14 (repo/ restructure + push)** — Dropped `repo/` from git; the deploy now
  ships only the 3-file delta (`upstream-delta/`: patches 066/067 + `common.sh.patch` + a
  README) and `install.sh` clones the upstream rig at pinned `9e533ae` and applies the
  delta (`step_rig`). Verified `upstream@9e533ae` + delta reproduces the live `repo/` tree
  byte-for-byte (full `diff -r`). README: added a Provenance section; restored the
  dual-3090 framing (the 4080 SUPER + 4060 Ti ~16 GB Ada config is buggy, so dual RTX 3090
  is the only validated config). Committed `77fee1c` (912 files: 907 `repo/` removals +
  `upstream-delta/` + `install.sh`/`.gitignore`/`README.md`) and **pushed** to
  `git@github.com:suedenglehem/qwen3.8-27b-sglang-on-dual-rtx3090.git`. `run_i7.sh` (the
  i7 4080/4060ti helper) left untracked.
