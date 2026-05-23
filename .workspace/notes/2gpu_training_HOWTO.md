# 2-GPU Training HOWTO

## System requirements verified (RTX PRO 6000 Blackwell Max-Q × 2)

| Component | Required | Installed |
|---|---|---|
| OS | Ubuntu 24.04 LTS | Ubuntu 24.04.4 LTS |
| Kernel | ≥ 6.11 | 6.17.0-29 |
| Driver | ≥ 575.51 | 580.159.03 |
| CUDA toolkit | 12.9 | 12.9.2 |
| NCCL | bundled via pip | `nvidia_nccl_cu12-2.27.5` (no system package needed) |
| JAX | ≥ 0.5.3 | 0.6.2 |

Verify with:
```bash
nvidia-smi                              # both GPUs visible, driver 575+
python -c "import jax; print(jax.devices())"           # [CudaDevice(0), CudaDevice(1)]
python -c "import jax; print(jax.local_device_count())"  # 2
```

---

## How 2-GPU training works

Brax PPO uses `jax.pmap` automatically across all visible CUDA devices — no code changes needed. With 2 GPUs:
- Each GPU runs `num_envs / 2` parallel environments
- Gradients are averaged across GPUs via NCCL AllReduce after each step
- Both GPUs should show ~100% utilisation during training

The process shows both GPUs in `nvidia-smi` under the same PID — this is correct and expected.

---

## Recommended training parameters for 2 GPUs

```python
num_envs = 4096          # 2048 per GPU — sweet spot for compile time vs sample quality
batch_size = 2048        # 1024 per GPU
num_minibatches = 32
unroll_length = 20       # default, works well
```

**Divisibility constraint:** `num_envs` must be divisible by `num_devices × num_minibatches` (i.e. `num_envs % 64 == 0`).

Do **not** use `num_envs=16384` — it causes 8+ hour XLA compilation with no real quality benefit over 4096.

---

## Required environment variables

Set in `train_jax_ppo.py` before any JAX operation:

```python
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
# PCIe-only (no NVLink) tuning for dual RTX PRO 6000 Blackwell
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
os.environ.setdefault("NCCL_MIN_NCHANNELS", "8")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

_XLA_AUTOTUNE_PATH = "/tmp/xla_autotune.pbtxt"
_xla_flags_extra = [
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_shard_autotuning=false",           # workaround for Blackwell XLA hang
    "--xla_gpu_triton_gemm_any=true",
    f"--xla_gpu_dump_autotune_results_to={_XLA_AUTOTUNE_PATH}",
]
if os.path.exists(_XLA_AUTOTUNE_PATH):
    _xla_flags_extra.append(
        f"--xla_gpu_load_autotune_results_from={_XLA_AUTOTUNE_PATH}"
    )
xla_flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = " ".join([xla_flags] + _xla_flags_extra).strip()
```

**Critical:** use `os.path.exists` guard on the load flag — XLA crashes with
`FAILED_PRECONDITION: Autotune results file does not exist` if you try to load
a file that hasn't been written yet.

---

## Compilation caching (eliminates the 1–2 hour first-run cost)

Two independent caching mechanisms are used together:

### 1. XLA autotune pbtxt (`/tmp/xla_autotune.pbtxt`)
Saves XLA's kernel benchmark results (which tile sizes / configs are fastest).
Written on first run via `--xla_gpu_dump_autotune_results_to`, loaded on
subsequent runs via `--xla_gpu_load_autotune_results_from`. **This eliminates
Phase 1 (kernel search).**

### 2. JAX persistent compilation cache (`~/.cache/jax_compilation_cache/`)
Saves the compiled XLA computation graphs (the actual GPU machine code).
Configured in `train_jax_ppo.py`:

```python
_JAX_CACHE_DIR = os.path.expanduser("~/.cache/jax_compilation_cache")
os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
```

**This eliminates Phase 2 (JIT graph compilation).**

### Cache is shape-specific
Both caches are keyed on tensor shapes. Changing any of these forces a full
recompile:
- `num_envs`, `unroll_length`, `batch_size`, `num_minibatches`
- Network hidden layer sizes
- Number of GPUs

Changing these does **not** recompile:
- `num_timesteps`, `num_evals`, `learning_rate`, `discounting`, `entropy_cost`

### Expected compile time (first run, 4096 envs, 2 GPUs)
- Phase 1 (autotune): ~1 min with cache, ~15–30 min cold
- Phase 2 (JIT): already fast at 4096 envs
- **First `reward=` appears within ~90 seconds** once both phases complete

---

## Diagnosing a stuck/slow run

| Observation | Meaning |
|---|---|
| GPU: 100% compute, 0% memory bandwidth | XLA compilation in progress (normal) |
| GPU: 100% compute, >0% memory bandwidth | Training running (normal) |
| GPU: 0% both | Process dead or idle |
| All CPU threads in `futex_do_wait` | JAX dispatched async GPU work, CPU waiting |
| No stdout output | Python stdout was block-buffered — fixed by `sys.stdout.reconfigure(line_buffering=True)` |

Check the real training PID (not the shell wrapper):
```bash
ps aux | grep train-jax-ppo | grep -v grep
cat /proc/<PID>/wchan        # what syscall is it in
cat /proc/<PID>/status | grep VmRSS  # memory usage
```

---

## Known issue: IOMMU/ACS causing silent NCCL P2P failure on Blackwell

**Symptom:** Training appears to run (GPUs at 100% compute) but 10M steps takes 40+ minutes instead of ~10.

**Root cause:** IOMMU is enabled on this system (53 IOMMU groups active). The two GPUs sit on separate PCIe buses (`c1:00.0` and `e1:00.0`, topology: `NODE`). With IOMMU + ACS active, NCCL cannot use direct P2P and silently falls back to a slow host-memory path for AllReduce gradient sync.

**GPU topology:**
```
        GPU0  GPU1
GPU0     X    NODE    ← NODE = PCIe via host bridge, no NVLink
GPU1    NODE    X
```

**Confirmed diagnostic result (2026-05-23):**

| Config | 1M env steps | Extrapolated 100M steps |
|--------|-------------|------------------------|
| No P2P disable (broken IOMMU path) | 40+ min (didn't finish) | > 4 hours |
| `NCCL_P2P_DISABLE=1` (host-memory AllReduce) | **~80 seconds** | ~2.2 hours |
| After BIOS fix (direct PCIe P2P) | estimated | ~25–35 min |

**Quick diagnostic:**
```bash
NCCL_P2P_DISABLE=1 train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 1000000 --num_evals 1
```
If this is significantly faster than a run without the flag, IOMMU/P2P is the bottleneck.

**Fix options (in order of preference):**
1. **BIOS fix (permanent, best):** Disable ACS in BIOS + add `iommu=pt pci=realloc` to kernel cmdline (`/etc/default/grub`). See `bios_fix.md` for exact steps. ~4-6× faster than option 2.
2. **Software workaround:** `NCCL_P2P_DISABLE=1` — prevents hangs, correct training, but slower (~2.2 hrs for 100M steps).

**Note:** Until BIOS is fixed, single-GPU training with `CUDA_VISIBLE_DEVICES=0` and `num_envs=8192` may be faster than broken 2-GPU training.

---

## What NOT to do

- **Do not use `num_envs=16384`** — causes 8+ hour compile, possibly infinite
- **Do not add `--xla_gpu_load_autotune_results_from` without checking the file exists** — hard crash
- **Do not kill a run mid-compile expecting the cache to be valid** — partial cache files will be ignored on next run (bad hash), forcing full recompile
- **Do not use `nohup ... > log 2>&1` without `sys.stdout.reconfigure(line_buffering=True)`** — Python block-buffers stdout in non-interactive mode and you'll see no output until the process exits
- **Do not add `--xla_gpu_enable_async_collectives=true` to XLA_FLAGS** — not a valid flag in JAX 0.6.2, causes immediate XLA abort with "Unknown flag" error
