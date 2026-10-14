# upstream-delta/ — the only thing this deploy adds on top of the upstream rig

This deploy is **mattbucci's 2×3090 sglang inference rig, plus three small changes**.
Rather than vendoring the whole rig into this repo, we store just the delta here and
have `install.sh` clone the upstream at a pinned commit and apply it.

## Upstream

- **URL:** <https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference>
  (also in `../url.txt`)
- **Pinned commit:** `9e533aecc77bff8f8546824fcd2c12a1c370f45b`
  (2026-10-12, "README bake-off table…"). This is the exact commit the live rig was
  cloned from: `upstream@9e533ae` + the three files below reproduces the running
  `repo/` tree **byte-for-byte** (verified by a full `diff -r`). The next commit on
  `main` (`b681e85`) only advances benchmark docs and is *not* what this rig was
  built from, so we pin to `9e533ae` to keep a fresh install identical to the
  production one.

## The three delta files

| File | What it is |
|---|---|
| `066-anthropic-effort-passthrough.patch` | **new** sglang patch — passes the Anthropic `reasoning_effort` through so Claude Code's effort ladder reaches the server |
| `067-server-default-sampling-params.patch` | **new** sglang patch — server-wide default sampling params (see README "What every request gets") |
| `common.sh.patch` | **modifies** `scripts/common.sh` — the conda-free guard (`SGLANG_NO_CONDA`): proceed when a venv is active instead of aborting on "Cannot find conda" |

Patches `001`–`065` and every other rig file (`scripts/launch.sh`, docs, …) come
straight from upstream — they are **not** part of the delta.

## How `install.sh` uses this

`step_rig` (in `install.sh`):

1. `git clone <URL> repo/` (or re-pin an existing clone) and `git reset --hard 9e533ae`.
2. `cp 066-*.patch 067-*.patch repo/patches/`
3. `git -C repo apply common.sh.patch`

Result: `repo/` = upstream@9e533ae + delta = the 33-patch rig this deploy serves.
`repo/` itself is gitignored — it is rebuilt from upstream + this delta on every install.
