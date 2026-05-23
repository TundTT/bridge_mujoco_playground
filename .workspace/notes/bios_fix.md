# BIOS Fix: Enable NCCL P2P for Dual RTX PRO 6000 Blackwell

## Problem

With IOMMU + ACS enabled, the two GPUs cannot use direct PCIe P2P for NCCL
AllReduce (gradient sync). NCCL silently falls back to routing through host
RAM, causing 2-GPU training to be 4–10× slower than expected.

---

## Fix 1 — BIOS changes (do this first)

### Step 1: Disable ACS in BIOS
ACS (Access Control Services) is a PCIe feature that blocks direct GPU-to-GPU
transfers for IOMMU isolation. It must be disabled for NCCL P2P to work.

- Reboot into BIOS (usually Delete or F2 at POST)
- Navigate to: **Advanced → PCIe/PCI Configuration** (exact path varies by board)
- Find **ACS** or **Access Control Services** — set to **Disabled**
- Also check for **SR-IOV** — disable if present (can re-enable ACS implicitly)
- Save and exit

> Board-specific notes:
> - **ASUS WRX90E SAGE SE**: Advanced → AMD CBS → NBIO Common Options → ACS Enable → Disabled
> - **Supermicro WRX90**: Advanced → PCIe/PCI/PnP → ACS Control → Disabled
> - **ASRock WRX90**: Advanced → Chipset → South Bridge → ACS Enable → Disabled

---

## Fix 2 — Kernel cmdline (do this after BIOS change)

Even with ACS disabled in BIOS, the Linux IOMMU driver may still enforce
isolation. Set IOMMU to passthrough mode to allow direct P2P:

### Step 1: Edit GRUB
```bash
sudo nano /etc/default/grub
```

Find the line:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```

Change to:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt pci=realloc"
```

- `iommu=pt` — passthrough mode: IOMMU stays on for DMA protection but allows
  direct P2P between devices
- `pci=realloc` — reallocates PCIe BARs to ensure both GPU memory windows are
  accessible across the root complex

### Step 2: Update GRUB and reboot
```bash
sudo update-grub
sudo reboot
```

---

## Verification after reboot

```bash
# Confirm kernel cmdline took effect
cat /proc/cmdline | grep iommu

# Test NCCL P2P works
cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate
python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True)
print(result.stdout)
"
# Should still show NODE (PCIe topology doesn't change) but NCCL will now
# use P2P transfers rather than host-memory fallback.

# Quick training speed test — should reach 1M steps in < 2 minutes
NCCL_DEBUG=INFO train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 1000000 --num_evals 1 2>&1 | grep -E "reward=|NCCL|transport"
# Look for "NCCL INFO ... via P2P" in output confirming P2P is active
```

---

## Fallback (if BIOS fix is not possible)

If you cannot modify BIOS (e.g., locked workstation), use the software
workaround. It prevents hangs but is slower than true P2P:

```bash
export NCCL_P2P_DISABLE=1
```

Or permanently in `train_jax_ppo.py`:
```python
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
```

---

## Expected result after fix

Confirmed baseline (2026-05-23) on this machine:

| Config | Timing |
|---|---|
| 2-GPU, no fix (broken IOMMU P2P) | 40+ min for 1M steps, never finished |
| 2-GPU, `NCCL_P2P_DISABLE=1` workaround | ~80 sec / 1M steps → ~2.2 hrs / 100M steps |
| 2-GPU, after BIOS fix (estimated) | ~20–30 sec / 1M steps → ~25–35 min / 100M steps |

---

## References
- NVIDIA Developer Forums: NCCL P2P hang on dual RTX PRO 6000 Blackwell WRX90E
- Level1Techs: Dual RTX PRO 6000 Blackwell Max-Q — how to make P2P/NCCL work
- NVIDIA JAX Release Notes 25.01: Blackwell-specific XLA issues
