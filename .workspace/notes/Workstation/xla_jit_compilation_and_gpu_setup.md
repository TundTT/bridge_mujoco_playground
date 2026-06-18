# Training Lessons Learned

## XLA JIT Compilation

### Do not override `--num_envs` without adjusting batch math
The PPO config for `Go1BridgeCrossing` is designed around `num_envs=8192`.
Passing `--num_envs 512` creates minibatch shapes of 320 (instead of 5120) that
are not in XLA's GEMM hint library. XLA falls back to exhaustive autotuning and
can hang for 3+ hours without producing any training output.

**Rule:** always use the config's default `num_envs`, or adjust `batch_size` and
`num_minibatches` together to keep minibatch size = `num_envs * unroll_length / num_minibatches`
at a standard shape (power-of-2 thousands work well).

### XLA autotuning warnings are a leading indicator
```
All configs were filtered out... Working around this by using the full hints set instead.
```
This warning means XLA is doing exhaustive kernel search for that matrix shape.
A few warnings at startup are normal. If they keep firing after the first minute,
the run will likely hang for hours.

---

## Video Rendering

The training script sets `MUJOCO_GL=egl` internally, but this does not always
take effect for the renderer launched after training. Set it explicitly in the shell:

```sh
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl train-jax-ppo --env_name Go1BridgeCrossing --use_wandb
```

---

## WandB

- Project: `bridge_crossing`, entity: `Tund`
- All per-reward-term metrics flow automatically via `state.metrics["reward/{k}"]`
  → brax training wrapper → `eval/episode_reward/{k}` in WandB
- WandB connects and logs correctly before training begins; confirmed by initial
  eval metric appearing at step 0

---

## Recommended Smoke Test Command

```sh
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl train-jax-ppo \
  --env_name Go1BridgeCrossing \
  --num_timesteps 1000000 \
  --num_evals 2 \
  --use_wandb
```

Expected: ~60s compile, ~70s training, 2 reward data points in WandB.

## Recommended Full Training Command

```sh
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl train-jax-ppo \
  --env_name Go1BridgeCrossing \
  --use_wandb
```
