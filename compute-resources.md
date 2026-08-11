# Compute Resources

Inventory of every compute resource reachable from this workstation, as of **2026-08-11**.
All hardware/quota figures below were queried live on that date unless marked otherwise —
balances and instance states drift, so re-run the commands in
[§6 Re-checking live state](#6-re-checking-live-state) before planning a large run.

---

## At a glance

| Resource | Scale | Access | Status (2026-08-11) |
|---|---|---|---|
| **PSC Bridges-2** — `ROBO` partition | 6 nodes × 8× H100 (2 TB RAM, 96 cores each) | `ssh bridges2`, Slurm | Available; dedicated 1-node reservation until **Aug 15** |
| **PSC Bridges-2** — `GPU` partition | 46 nodes: 10× (8× H100-80), 3× (8× L40S-48), 33× (8× V100-32) | `ssh bridges2`, Slurm | 20,059 SU left of 148,428 |
| **PSC Bridges-2** — CPU (`RM`/`EM`) | 484 CPU nodes; 4 extreme-mem nodes @ 4 TB | `ssh bridges2`, Slurm | Nearly exhausted: 5,816 SU of 488,000 |
| **AirLab Cloud** (OpenStack) | Up to 4× A100 per instance (quota-capped) | `openstack` CLI + CMU VPN | 160/200 vCPU consumed by an **ERROR'd instance** |
| **4090 dev box** (`hshi4090`) | 1× RTX 4090 24 GB, 32 cores, 62 GB RAM | `ssh hshi4090` (Tailscale) | Up |
| **Local Mac** | Apple M1 Max, 10 cores, 32 GB unified | local | Up |

---

## 1. PSC Bridges-2 — primary training cluster

Largest resource by far, and where all v2 8B production training runs.

- **Host:** `bridges2.psc.edu` — `ssh bridges2` (alias in `~/.ssh/config`, with
  `ControlMaster` so one password login is reused by later sessions)
- **User:** `hshi6`
- **Account:** `cis220039p` — *Dense 3D Reconstruction of Dynamic Actors in Natural Environments*
- **Project root:** `/ocean/projects/cis220039p/hshi6/tartanimu-dev-sdft`

### 1.1 Allocation balances

All four grants expire **2026-09-30**. Values are *remaining / total*.

| Grant | Remaining | Total | Note |
|---|---:|---:|---|
| Bridges-2 **Regular Memory** (CPU) | 5,816 SU | 488,000 SU | ~1% left — effectively spent |
| Bridges-2 **GPU** | 20,059 SU | 148,428 SU | ~14% left — the real GPU budget |
| **Robo GPU** | −222,377 SU | 0 SU | No SU grant; usage tracked negative. ROBO jobs still schedule and run — the `ROBO` partition appears not to be SU-gated for this account, but this is inferred from behavior, not confirmed with PSC. |
| `/ocean/projects` storage | 745,317 GB | 1,839,248 GB | Shared across the whole project group |

### 1.1b Storage layout

| Path | Size | Used | Purpose |
|---|---:|---:|---|
| `/ocean/projects/cis220039p` | 1,839,248 GB granted | 745,317 GB remaining | Group-wide project space |
| `/ocean/projects/cis220039p/hshi6` | — | **1.1 TB** | Your subtree: repo, `ckpts/`, `data_cache_v2`, `hf_cache`, `rollouts/`, `logs/` |
| `/jet/home/hshi6` | 25 GB | 449 MB (2%) | Home. Small by design — never put checkpoints or HF caches here (batch scripts already redirect `HF_HOME`/`TMPDIR`/caches to `/ocean`) |

### 1.2 Partitions and hardware

Verified with `sinfo -s` and `scontrol show node` on 2026-08-11.

| Partition | Nodes | GPUs/node | Cores/node | RAM/node |
|---|---|---|---|---|
| `ROBO` | `robo-gh[001-006]` (6) | 8× H100 | 96 (2×48) | 2,063,730 MB (~2.0 TB) |
| `GPU`, `GPU-shared` | `w[001-010]` (10) | 8× H100-80 | 104 | 2,063,900 MB |
| | `gl[001-003]` (3) | 8× L40S-48 | 192 | 1,030,000 MB |
| | `v[002-034]` (33) | 8× V100-32 | 40 | 515,000 MB |
| `GPU-small` | `v001` (1) | 8× V100-32 | 40 | 515,000 MB |
| `GPU-dev` | `a001` (1) | 1× A100 | 48 | 515,000 MB |
| `RM`, `RM-shared` | `r[005-488]` (484) | — | 128 | 256,000 MB |
| `RM-small` | `r[001-004]` (4) | — | 128 | 256,000 MB |
| `RM-512` | `l[002-016]` (15) | — | 128 | 515,000 MB |
| `EM` (extreme memory) | `e[001-004]` (4) | — | 96 (4 sockets) | 4,128,000 MB (~4 TB) |
| `HACC` | `hacc-gm[001-005]` (5) | 4× AMD MI210 | 192 | 754,000 MB |
| `applications` | `l001` | — | — | — |

Notes:
- Despite the `-gh` hostnames, the ROBO nodes report `Arch=x86_64` and `Gres=gpu:h100:8` —
  they are x86 + H100, **not** Grace-Hopper GH200.
- All 5 `HACC` (MI210) nodes were in a down/drained state on 2026-08-11 — treat as unavailable.
- No time limit is configured on the compute partitions (`TIMELIMIT=infinite`), but jobs
  longer than the 5-day backfill window cannot backfill. Project convention caps walltime at
  **4d20h (116h)** — see `scripts/psc/autopilot_v2.sh`.

### 1.3 QOS available to `cis220039p`

`ft, gpu, gpuinteract, low, push, rm, rminteract, robo, robointeract, unlimited`

`--qos=push` is used for smoke jobs to jump the queue; production runs on the default ROBO QOS.

### 1.4 Reservation (time-limited)

```
ReservationName=ROBOcis220039p
Nodes=robo-gh006  NodeCnt=1  CoreCnt=96  PartitionName=ROBO
2026-07-31T12:00 → 2026-08-15T12:00   State=ACTIVE   Accounts=cis220039p
```

A whole 8× H100 node held for the account, **expiring 2026-08-15**. Jobs opt in with
`#SBATCH --reservation=ROBOcis220039p` (used by `retention_backfill.sbatch` and
`rollout_init.sbatch`) so they never compete with other users' general-queue jobs.
Anything that depends on the reservation must land before Aug 15 or be re-planned
onto the general ROBO queue.

### 1.5 Environment and gotchas

- **No working `rsync` on the login nodes** (no binary on `PATH`; the
  `/opt/packages/rsync/3.2.3` copy is missing `libxxhash`). Sync with tar-over-ssh via
  [`scripts/sync_psc.sh`](../scripts/sync_psc.sh).
- Python env is a self-contained conda env at `$ROOT/conda-env` — batch scripts prepend it to
  `PATH` rather than using `conda activate` shell hooks.
- `CUDA_HOME=/opt/packages/cuda/v12.6.1` (newest 12.x with `nvcc`; torch is cu128).
- Multi-GPU training runs through `accelerate` + DeepSpeed ZeRO-3
  (`scripts/psc/zero3.yaml`); single-process + `device_map` is not enough at 7B/8B.
- Co-scheduled jobs on one node must not both bind torch-elastic's default port 29500 —
  templates derive `MASTER_PORT` from `SLURM_JOB_ID`.
- Caches are redirected into the project dir: `HF_HOME`, `WANDB_DIR`, `TMPDIR`,
  `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`.

---

## 2. AirLab Cloud (CMU OpenStack)

Self-service VMs with passthrough GPUs. Secondary to PSC; useful when Slurm queues are long
or a long-lived interactive box is wanted.

- **Endpoint:** `https://airlab-cloud.andrew.cmu.edu:5000` (region `Airlab`);
  web UI at `https://airlab-cloud.andrew.cmu.edu`
- **Project:** `tartanstar.AirLab.Apps_group_project` (`a1daa88e6a104eb2bddac60a7be8ecc5`)
- **Credentials:** `~/.airlabcloud/app-cred-chris-cli-openrc.sh` — a self-contained
  application credential, no password prompt. `source` it, then use the `openstack` CLI.
- **Network access:** floating IPs live on `172.19.220.0/24`, which is **not routable without
  CMU VPN / campus network**. A failed ping there is a client-side gap, not a dead instance.

### 2.1 Quota and current usage

| Item | Used | Quota |
|---|---:|---:|
| vCPU | 160 | 200 |
| RAM | 614,400 MB | 716,800 MB |
| Instances | 2 | 20 |
| Volumes | 5 | 40 |
| Volume storage | 2,099 GB | 10,000 GB |
| Backup storage | 0 GB | 40,000 GB |

**Only 40 vCPU / 102,400 MB of headroom right now** — not enough for any multi-GPU flavor.

### 2.2 Instances

| Name | Flavor | Status | IPs |
|---|---|---|---|
| `refinement1` | `gpu.a100.1` | ACTIVE | 10.0.180.16 / 172.19.220.58 |
| `sdft-8b-a100x4` | `gpu.a100.4` | **ERROR** | — |

> ⚠️ `sdft-8b-a100x4` is in `ERROR` but still **consuming its full 128 vCPU / 480 GB of quota**
> and holding the `sdft-8b-boot` volume in `reserved` state. That is exactly what is blocking
> the `poll_a100x4.sh` poller (see §2.5). Deleting it would free 128 vCPU + the boot volume;
> this document does not do that.

### 2.3 Volumes

| Name | Size | Status |
|---|---:|---|
| `sdft-8b-data` | 1000 GB | available |
| `sdft-8b-boot` | 250 GB | reserved (held by the ERROR'd instance) |
| `tartanimu-dev-data` | 500 GB | available |
| `tartanimu-dev-sys` | 150 GB | available |
| *(unnamed)* | 199 GB | in-use |

Backing store is `gates-rbd`; boot image `Ubuntu-24.04-GPU-Headless`; keypair `chrisshi-mac`;
network `tartanstar.AirLab.Apps_group_network_gates`.

### 2.4 GPU flavors

| Flavor | vCPU | RAM (MB) | Disk (GB) | Launchable under quota? |
|---|---:|---:|---:|---|
| `gpu.a100.1` | 32 | 122,880 | 100 | yes |
| `gpu.a100.2` | 64 | 245,760 | 100 | yes |
| `gpu.a100.4` | 128 | 491,520 | 100 | yes — **largest A100 flavor available** |
| `gpu.a100.8` | 128 | 983,040 | 100 | **never** — RAM exceeds the project's *total* quota |
| `gpu.rtx6000ada.1` | 42 | 163,840 | 100 | yes |
| `gpu.rtx6000ada.2` | 84 | 327,680 | 100 | yes |
| `gpu.rtx6000ada.3` | 128 | 491,520 | 100 | yes |
| `gpu.rtx6000ada.6` | 256 | 983,040 | 100 | **never** — 256 vCPU > 200 vCPU quota |
| `gpu.rtxpro5000.1–4` | 38–48 | 56,320–225,280 | 100 | yes |
| `gpu.p100.1–8` | 20–80 | 56,320–450,560 | 100 | yes |

Key constraint: **`gpu.a100.8` can never launch** — 983,040 MB exceeds the project's entire
716,800 MB RAM quota, so "No valid host" there is a quota block, not a capacity block. Always
subtract current usage from quota before probing flavors.

To unlock 8× A100, an admin must raise the project RAM quota to ≥1 TB via
Infrastructure → Identity → Projects → Modify Quotas. Contacts: Yaoyu Hu
<yaoyuh@andrew.cmu.edu>, or Basti / Wenshan.

### 2.5 Standing automation — **stopped 2026-08-11**

`~/.airlabcloud/poll_a100x4.sh` will grab a `gpu.a100.4` instance the moment two gates clear:
(1) `sdft-8b-boot` becomes `available`, and (2) quota headroom reaches 128 vCPU / 491,520 MB.
It never deletes anything, so it stayed blocked behind the ERROR'd instance for its whole
4d22h run. Stopped on 2026-08-11. Restart with:

```bash
INTERVAL=300 nohup ~/.airlabcloud/poll_a100x4.sh >/dev/null 2>&1 &
```

Pointless to restart until `sdft-8b-a100x4` is deleted — see §2.2.

---

## 3. The 4090 dev box

Fast-iteration box for data prep, smoke tests, and small-model work.

- **Access:** `ssh hshi4090` → `100.74.154.39` (Tailscale), user `hshi-j-4090`,
  hostname `hongyi-J`
- **GPU:** 1× NVIDIA GeForce RTX 4090, 24,564 MiB, driver 535.309.01
- **CPU/RAM:** 32 cores, 62 GB (≈52 GB available)
- **Disk:** 3.5 TB on `/`, **770 GB free (77% used)** — worth watching before large checkpoint runs
- **Working copy:** `~/SDFT-Repro`, pushed from the Mac by
  [`scripts/sync_4090.sh`](../scripts/sync_4090.sh)

> ⚠️ **Clobber hazard.** `sync_4090.sh` uses `rsync --delete` from the Mac's *main* checkout and
> excludes only `data_cache/`, `ckpts/`, `results/`. Multiple sessions share this one execution
> copy, so a sync from another session can silently revert `src/` and `tests/` mid-run (this
> happened on 2026-08-05 and produced a bogus 0.0 eval). When running worktree-branch code here:
> re-sync immediately before launching, verify the entry point's imports resolve on the box,
> treat surprising all-zero results as possible clobber (a reverted file carries an *old* mtime),
> and merge branches to main promptly.

---

## 4. Local workstation

- **Machine:** Apple M1 Max, 10 cores, 32 GB unified memory
- **Disk:** 926 GB volume, ~147 GB free
- **Role:** source of truth for the repo (`/Users/hshi/Desktop/SDFT-Repro`), orchestration,
  analysis of pulled-back results (`data_cache_from_4090/`, `data_cache_v2_from_4090/`,
  `results/`). Not used for training.

---

## 5. Known but unavailable

- **`gpu.a100.8` / `gpu.rtx6000ada.6`** on AirLab — permanently quota-blocked (§2.4).
- **`HACC` MI210 nodes** on Bridges-2 — all 5 down/drained as of 2026-08-11.

Out of scope: the `shark` host in `~/.ssh/config` (`nurseshark.ics.cs.cmu.edu`) is a
coursework machine, not a research resource. Ignore it.

---

## 5b. Background automation — all stopped 2026-08-11

Nothing is polling any of these resources right now. Both long-running loops lived on the
**Mac**, not on PSC or the 4090.

| Loop | Ran | Last useful activity | Restart |
|---|---|---|---|
| `~/.airlabcloud/poll_a100x4.sh` | Aug 6 13:52 → Aug 11, 4d22h | Never fired — blocked on the ERROR'd instance the entire time | §2.5 |
| `scripts/watcher/watch_checkpoints.sh` | Aug 7 00:01 → Aug 11, 4d12h | **Aug 7 18:15** (`sft seg1-tooluse ckpt_step00220`); log went silent **Aug 10 08:28** — the loop was hung, not idle | header of the script |

On PSC, `autopilot_v2.sh` completed on its own (Aug 6 17:19:57, `=== autopilot_v2 done`); no
detached processes remain under `hshi6` on the login node checked (`br014`), and `squeue` is
empty.

The 4090 is running `sdpo-diffing/scripts/recovery2.py` — a **different project's** job, left
untouched. Note that the watcher's `gpu_busy()` guard only greps for `lm-eval`/`vllm`, so it
would *not* have detected that job and could have contended with it for the 24 GB card.

---

## 6. Re-checking live state

```bash
ssh bridges2 'projects; squeue -u $USER; sinfo -p ROBO -o "%20N %10t %10G %10m"; scontrol show res'
```

```bash
source ~/.airlabcloud/app-cred-chris-cli-openrc.sh && openstack server list && openstack limits show --absolute
```

```bash
ssh hshi4090 'nvidia-smi; df -h ~; free -g'
```
