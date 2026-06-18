# Training Code Structure

Three files are involved when running `train-jax-ppo --env_name Go1BridgeCrossing`.
They have distinct responsibilities and are not the same file.

## `learning/train_jax_ppo.py` — the training runner

Entry point for the `train-jax-ppo` CLI command. Generic across all environments.
Responsibilities:
- CLI flags (`--env_name`, `--use_wandb`, `--num_timesteps`, etc.)
- GPU setup: `CUDA_VISIBLE_DEVICES`, `MUJOCO_GL`, XLA flags
- WandB and TensorBoard initialisation
- Building and running the Brax PPO training loop
- Checkpointing and rollout video rendering

## `mujoco_playground/config/locomotion_params.py` — PPO hyperparameters

Maps environment names to tuned PPO configs. The runner calls
`get_rl_config("Go1BridgeCrossing")` at startup to get these values.
Responsibilities:
- `num_envs`, `batch_size`, `num_minibatches`
- Network architecture (`policy_hidden_layer_sizes`, `value_hidden_layer_sizes`)
- Learning rate, discounting, entropy cost, etc.

Go1BridgeCrossing block is around line 151.

## `mujoco_playground/_src/locomotion/go1/bridge.py` — the environment

The actual task implementation. Neither of the above files imports it directly;
they reference it by name through the registry.
Responsibilities:
- Physics setup, XML loading, bridge width patching
- `reset()`, `step()`, observations, rewards, termination

## Flow

```
train-jax-ppo --env_name Go1BridgeCrossing
      │
      ├── train_jax_ppo.py        (GPU setup, WandB, runs PPO loop)
      ├── locomotion_params.py    (PPO hyperparameters for this env)
      └── go1/bridge.py           (the environment being trained on)
```

GPU configuration (XLA flags, CUDA_VISIBLE_DEVICES) lives in `train_jax_ppo.py`,
not in the PPO hyperparameter config.
