# Qwen3.8-27B on sglang — dual RTX 3090

Serves **Qwen3.8-27B** on two RTX 3090 (24 GB each) with a patched **sglang v0.5.20** tree.
One entry point for everything: **`./run_qwen38.sh`**.

Built on [mattbucci's 2×3090 sglang rig](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference)
(a pinned commit + a 3-file delta — see "Provenance" below), adapted to a conda-free host
(uv venv) and to Qwen3.8-27B. **`PROGRESS.md` is the knowledge base** — read it for the
*why* (commit pin, venv recipe, memory model, GPU-mapping trap, template rule, measured
numbers). This file is the *how*.

On the author's host (x399 motherboard / PCIe gen3, Threadripper 1950x, 64 GB DDR4-3600),
the **FP8 (q8) variant reaches ~80 tok/s** — about **10 tok/s more than vLLM** on the same
hardware and variant.

## Provenance: the upstream rig

This deploy is [mattbucci's 2×3090 sglang inference rig](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference)
with three small additions. Rather than vendoring the whole rig, this repo stores just the
delta (`upstream-delta/`) and `install.sh` clones the upstream at a pinned commit and applies
it:

- **Upstream:** <https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference> (also in `url.txt`)
- **Pinned commit:** `9e533ae` — the exact commit the rig was built from; `upstream@9e533ae`
  + the delta reproduces the running `repo/` tree byte-for-byte.
- **The delta** (`upstream-delta/`): patches `066` (Anthropic effort passthrough) and `067`
  (server-default sampling) plus a `common.sh` patch (the conda-free guard). Everything else —
  patches `001`–`065`, `scripts/launch.sh`, docs — comes straight from upstream.

See `upstream-delta/README.md` for the full provenance.

## Requirements

- 2× NVIDIA RTX 3090 (24 GB) — the only validated config (TP=2). Keep the two cards
  the same size class (the launcher checks that they balance in memory).
  - **INT4** (AWQ) is the default variant; **FP8** costs ~2× the VRAM of INT4.
- NVIDIA driver (recent; CUDA 13 runtime is pulled by the torch wheels, no toolkit needed)
- Internet (clones sglang, builds the venv, downloads the model)
- Disk: ~35 GB free for the FP8 variant (31 GB weights + venv + JIT cache), ~25 GB for INT4
- Linux, bash, git, curl

## Quick start

```bash
git clone git@github.com:suedenglehem/qwen3.8-27b-sglang-on-dual-rtx3090.git
cd qwen3.8-27b-sglang-on-dual-rtx3090
./install.sh fp8          # or: ./install.sh int4
./run_qwen38.sh -fp8 -mtp # serve FP8 + MTP speculative decoding on port 8091
```

`install.sh` clones the upstream rig + applies our delta, clones + patches sglang, builds the
venv, fetches the model (prompts for a predownloaded path or pulls from Hugging Face), writes
`config.env`, and verifies the install. First boot after install compiles the JIT kernel cache
(~12 min); subsequent boots are ~55–70 s.

## Model variants

| Flag | Checkpoint | Quant | KV pool (TP=2) | Notes |
|---|---|---|---|---|
| *(no flag)* or `-i4` | [hf-mattbucci/Qwen3.8-27B-AWQ](https://huggingface.co/hf-mattbucci/Qwen3.8-27B-AWQ) | awq_marlin (INT4) | ~697K tokens | the default; no MTP weights |
| `-fp8` | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | fp8 (e4m3, dynamic act) | ~259K tokens | official Qwen release; **full VLM** (vision+video in the checkpoint) |
| `-fp8 -mtp` | same as `-fp8` | fp8 + MTP draft | 165,532 tokens | ~2.5–3× decode speedup; FP8-only |

Model sources (Hugging Face): INT4 → <https://huggingface.co/hf-mattbucci/Qwen3.8-27B-AWQ>,
FP8 → <https://huggingface.co/Qwen/Qwen3.8-27B-FP8>.

Same model (`qwen3_5`, `Qwen3_5ForConditionalGeneration`) in two quantizations. FP8 weights
cost ~2× the VRAM of INT4, so its pool only just covers the 262144 context claim — deep
requests past the pool fail on that variant (see `PROGRESS.md` §7–8 for the memory model).

## Run flags

```bash
./run_qwen38.sh                 # start (foreground; Ctrl-C stops) — INT4 default
./run_qwen38.sh -i4             # AWQ int4 ship (explicit spelling of the default)
./run_qwen38.sh -fp8            # FP8 checkpoint
./run_qwen38.sh -fp8 -mtp       # FP8 + MTP speculative decoding
./run_qwen38.sh --all-gpus      # perf-test: all 3 cards incl. any display card (PP=3)
./run_qwen38.sh -fp8 -mtp --print-cmd   # print the server command as one line (pass last)
DRY_RUN=1 ./run_qwen38.sh [flags]       # print the exact argv without launching
```

Flags compose freely (`-fp8 -mtp --all-gpus`). Extra flags pass through to
`repo/scripts/launch.sh` (e.g. `--port 30001`, CTX/MEM overrides).

**`--all-gpus` is a perf-test exception, never the default** — it pulls in a third (display)
card and runs PP=3, which is slower than TP=2 (pipeline bubbles; the slowest card gates
every step; the pool is a quarter the size). See `PROGRESS.md` §11.

### Detached run / stop

```bash
setsid bash -c './run_qwen38.sh -fp8 -mtp' > server.log 2>&1 & echo $!   # note the PID
kill <PID>          # sglang tears down its children; confirm with nvidia-smi (~0 MiB)
pgrep -af 'sglang.launch_serve[r]'   # find a running server
```

## GPU selection (verify per host)

`run_qwen38.sh` defaults to `CUDA_VISIBLE_DEVICES=0,1` (the two 24 GB cards) and honors an
existing `CUDA_VISIBLE_DEVICES`. **CUDA index order is host-specific** and does not always
match `nvidia-smi` order. Before first launch on a new machine, verify which indices are
your two 24 GB cards:

```bash
python -c "import torch; [print(i, torch.cuda.get_device_name(i),
    torch.cuda.get_device_properties(i).total_memory/2**30, 'GB')
    for i in range(torch.cuda.device_count())]"
```

then set `CUDA_VISIBLE_DEVICES` (or the commented line in `config.env`) accordingly. The
full trap, with this host's verified mapping, is in `PROGRESS.md` §5.

## What every request gets

Server-wide default sampling (applies **only where a request omits the value** — explicit
client params always win):

```json
{"temperature": 0.6, "top_p": 0.95, "top_k": 20,
 "presence_penalty": 0.0, "frequency_penalty": 0.0,
 "repetition_penalty": 1.0, "max_new_tokens": 16384}
```

Covers `/v1/chat/completions` and `/v1/messages`. Override per launch with your own
`SGLANG_DEFAULT_SAMPLING_PARAMS='...'` env var.

**Chat template:** `templates/qwen3.8.q6.sglang.jinja`, passed **verbatim** — one block is
deliberately modified (the reasoning-effort ladder) and that is what makes Claude Code
accept the model. The original template is shared with other stacks and must never be
edited; see `PROGRESS.md` §6.

## Layout

| Path | What |
|---|---|
| `run_qwen38.sh` | launcher (self-locating; reads `config.env`) |
| `install.sh` | installer (int4 \| fp8) |
| `config.env` | host-specific paths (gitignored; written by `install.sh`) |
| `templates/` | the chat template we serve |
| `upstream-delta/` | the 3-file delta applied on top of the upstream rig (patches 066/067 + a `common.sh` patch) |
| `repo/` | the serving rig: upstream mattbucci @ pinned commit + our delta (built by `install.sh`) |
| `sglang-rebase-v0520/` | sglang v0.5.20 + patches (built by `install.sh`) |
| `v0520/` | uv venv (built by `install.sh`) |
| `models/` | downloaded weights (built by `install.sh`) |
| `PROGRESS.md` | the knowledge base |

## Further reading

- `PROGRESS.md` — stack recipe, memory model, traps, measured numbers
- `upstream-delta/README.md` — the upstream git, the pinned commit, and the 3-file delta
- `repo/patches/README.md` — per-patch narratives (from the upstream rig, present after install)
- `repo/CLAUDE.md`, `repo/rules-for-agents.md` — the upstream rig's rules (VRAM budget,
  launch rules, quantization)
