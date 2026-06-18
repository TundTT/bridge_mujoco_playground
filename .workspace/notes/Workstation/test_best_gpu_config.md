# TODO

## Experiment A — double envs on 2 GPUs
- Try `num_envs=8192, batch_size=4096` (double current config, 4096/GPU)
- Expect ~5–10 min compile on cold cache, normal range for 2-GPU pmap
- See `2gpu_training_HOWTO.md` for divisibility constraints and cache behaviour
- Only worth doing if current 4096-env run shows slow convergence

## Experiment C — push num_envs to saturate both GPUs
- Motivation: sync overhead is fixed per step — more envs = more useful work per sync = 2-GPU advantage grows
- Current VRAM usage: ~1.2GB / 97GB per card → massive headroom
- Step up incrementally, timing compile at each step:
  - 8192 envs (4096/GPU, batch_size=4096)
  - 16384 envs (8192/GPU, batch_size=8192)
  - 32768 envs (16384/GPU, batch_size=16384) if compile is still reasonable
- Compile should be faster than the original 16384 disaster now that XLA autotune pbtxt is saved
- Goal: find the sweet spot where 2-GPU throughput approaches ~2× single GPU
- JAX cache won't hit (new shapes), but autotune pbtxt transfers some kernel results

## Experiment B — single GPU benchmark
- Motivation: small network (512→256→128) + physics sim may mean pmap AllReduce overhead outweighs 2-GPU benefit
- Run: `CUDA_VISIBLE_DEVICES=0 train-jax-ppo --env_name Go1BridgeCrossing --num_timesteps 10000000 --num_evals 1 --use_wandb`
- Compare wall-clock time per 10M steps vs current 2-GPU run
- If single GPU is faster → run production training as `CUDA_VISIBLE_DEVICES=0` with `num_envs=8192`
- If 2-GPU is faster → proceed with Experiment A
