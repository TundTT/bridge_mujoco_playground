# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bridge crossing task for Go1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


_HM_X_MIN, _HM_X_MAX = -3.0, 7.0
_HM_Y_MIN, _HM_Y_MAX = -1.0, 1.0
_HM_CELL = 0.03
_HM_PATCH = 13
# Platform and bridge surface z (used to convert absolute foot z to relative clearance).
_TERRAIN_Z = 0.5

# linvel(3)+gyro(3)+gravity(3)+joint_angles(12)+joint_vel(12)+last_act(12)+x_progress(1)+foot_y(4)+trunk_y(1)+bridge_hw(1)
_PROPRIO_SIZE = 52


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=3,
      soft_joint_pos_limit_factor=0.95,
      # Curriculum: bridge y half-extent in metres (0.4 = 0.8 m wide).
      bridge_half_width=0.4,
      # Virtual half-width lower bound for per-episode foothold shaping.
      # Each episode samples virtual_hw ~ U[virtual_hw_min, bridge_half_width].
      # Reward and obs use virtual_hw; physical falls happen at bridge_half_width.
      virtual_hw_min=0.15,
      # Terminate if root z drops below this (0.3 m below the 0.5 m platform surface).
      fall_threshold=0.2,
      # Terminate with success bonus when robot reaches this x (halfway into Platform B).
      # Platform B spans x=4.0 to x=7.0; midpoint is 5.5.
      goal_x=5.5,
      # Target forward speed for the tracking reward (m/s). Reduce at narrow stages.
      target_speed=0.5,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
              foot_y=0.005,
          ),
      ),
      reward_config=config_dict.create(
          # Max foot clearance height for gait shaping (metres).
          max_foot_height=0.08,
          scales=config_dict.create(
              orientation=-5.0,
              alive=0.0,
              torques=-0.0002,
              # First-order action smoothness.
              action_rate=-0.01,
              # Second-order action smoothness (jitter penalty).
              action_accel=-0.01,
              # Joint acceleration penalty (rad/s²)².
              dof_acc=-2.5e-7,
              energy=-0.001,
              termination=-200.0,
              success=5000.0,
              # Velocity tracking: exp(-|v_xy - target|²/0.25) × foothold_quality.
              # Replaces frontier_delta — concave in velocity so smooth > surge.
              tracking_lin_vel=2.0,
              # Air-time reward capped at 0.25s (raised from 0.15s so healthy
              # trot swings earn positively; jitter <0.1s still pays a penalty).
              feet_air_time=2.0,
              lateral_deviation=-3.0,
              heading=-2.0,
              foot_off_bridge=-50.0,
              lin_vel_z=-1.0,
              ang_vel_xy=-0.1,
              feet_slip=-0.25,
              feet_clearance=-2.0,
              feet_height=-0.2,
              pose=0.5,
              dof_pos_limits=-1.0,
          ),
      ),
      impl="jax",
      naconmax=4 * 4096,
      njmax=40,
  )


class BridgeCrossing(go1_base.Go1Env):
  """Cross the narrow bridge from Platform A to Platform B."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.FEET_ONLY_BRIDGE_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    # Patch bridge width from config before training starts.
    bridge_id = self._mj_model.geom("bridge").id
    self._mj_model.geom_size[bridge_id, 1] = self._config.bridge_half_width
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._build_world_heightmap()
    self._post_init()

  def _build_world_heightmap(self) -> None:
    xs = np.arange(_HM_X_MIN, _HM_X_MAX + _HM_CELL / 2, _HM_CELL)
    ys = np.arange(_HM_Y_MIN, _HM_Y_MAX + _HM_CELL / 2, _HM_CELL)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    hw = self._config.bridge_half_width
    on_platform_a = (xx >= -3.0) & (xx <= 0.0) & (np.abs(yy) <= 1.0)
    on_bridge      = (xx >=  0.0) & (xx <= 4.0) & (np.abs(yy) <= hw)
    on_platform_b  = (xx >=  4.0) & (xx <= 7.0) & (np.abs(yy) <= 1.0)
    hm = (on_platform_a | on_bridge | on_platform_b).astype(np.float32)
    self._world_heightmap = jp.array(hm)

    half = (_HM_PATCH - 1) // 2
    idx = np.arange(_HM_PATCH) - half
    dx, dy = np.meshgrid(idx * _HM_CELL, idx * _HM_CELL, indexing='ij')
    self._lh_dx = jp.array(dx)
    self._lh_dy = jp.array(dy)

  def _get_local_heightmap(self, data: mjx.Data) -> jax.Array:
    robot_x = data.qpos[0]
    robot_y = data.qpos[1]

    forward = data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
    fwd_norm = jp.linalg.norm(forward[:2]) + 1e-6
    cos_yaw = forward[0] / fwd_norm
    sin_yaw = forward[1] / fwd_norm

    dx_world = cos_yaw * self._lh_dx - sin_yaw * self._lh_dy
    dy_world = sin_yaw * self._lh_dx + cos_yaw * self._lh_dy

    sx = robot_x + dx_world
    sy = robot_y + dy_world

    nx, ny = self._world_heightmap.shape
    xi = jp.clip(jp.round((sx - _HM_X_MIN) / _HM_CELL).astype(jp.int32), 0, nx - 1)
    yi = jp.clip(jp.round((sy - _HM_Y_MIN) / _HM_CELL).astype(jp.int32), 0, ny - 1)

    return self._world_heightmap[xi, yi].ravel()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )

    # Contact sensor addresses: 4 feet × 3 terrain surfaces.
    # sensor_bridge_feet.xml uses "floor" as the platform_a label.
    terrain_labels = ["floor", "bridge", "platform_b"]
    self._feet_terrain_sensor_adrs = jp.array([
        [
            self._mj_model.sensor_adr[
                self._mj_model.sensor(f"{geom}_{label}_found").id
            ]
            for label in terrain_labels
        ]
        for geom in consts.FEET_GEOMS
    ])

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Small x/y jitter within platform_a footprint.
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

    # Small yaw jitter (±30°) so the policy learns to correct its heading.
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-0.5, maxval=0.5)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Sample virtual half-width for this episode: U[virtual_hw_min, bridge_half_width].
    # Clamp min ≤ max to guard against stage configs where hw < virtual_hw_min.
    # Reward and obs use this value; physical falls still happen at bridge_half_width.
    rng, key = jax.random.split(rng)
    vhw_min = min(self._config.virtual_hw_min, self._config.bridge_half_width)
    virtual_hw = jax.random.uniform(
        key, (),
        minval=vhw_min,
        maxval=self._config.bridge_half_width,
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "last_qvel": jp.zeros(self.mjx_model.nv),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "obs_history": jp.zeros((self._config.history_len - 1, _PROPRIO_SIZE)),
        # Frontier tracker: kept as a metric only (no longer drives reward).
        "max_x_reached": qpos[0],
        # Spawn x: used by step() to reset max_x_reached on episode boundary.
        "init_x": qpos[0],
        # Per-episode virtual bridge width for foothold shaping.
        "virtual_hw": virtual_hw,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    # Furthest x reached this episode (monotone). Sum/ep_len ≈ average frontier.
    metrics["metric/max_x_reached"] = jp.zeros(())
    # Forward velocity per step (useful alongside max_x_reached).
    metrics["metric/x_vel"] = jp.zeros(())
    # Touchdown count: episode sum = total first_contact events.
    # Normal trot ~100-150/ep; foot-tapping exploit ~500-1500/ep.
    metrics["metric/touchdown_count"] = jp.zeros(())
    # Air time accumulated at touchdown events. Divide by touchdown_count for mean.
    metrics["metric/air_time_sum_at_td"] = jp.zeros(())
    # Termination type: sum=1 if episode ended in fall, 0 if success or timeout.
    metrics["metric/term_fall"] = jp.zeros(())
    # Sum=1 if episode ended in success, 0 if fell or timed out.
    metrics["metric/term_success"] = jp.zeros(())
    # Fraction of steps where any abduction joint |action| > 0.95 (saturation proxy).
    metrics["metric/abduction_saturation"] = jp.zeros(())
    # Max |foot_y| across all feet per step. Div by ep_len = avg worst-case lateral.
    metrics["metric/max_foot_y"] = jp.zeros(())
    # Virtual bridge width used this episode (constant within episode).
    metrics["metric/virtual_hw"] = jp.zeros(())
    # Mean swing peak relative to deck (healthy trot ~0.08m; hopping if >0.15m).
    metrics["metric/swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    # Detect EpisodeWrapper timeout: AutoReset teleports the robot back to spawn,
    # so x drops by >0.5m in one ctrl step — impossible under normal physics.
    teleported = state.data.qpos[0] > data.qpos[0] + 0.5

    # True if any terrain surface is under that foot.
    contact = jp.any(
        data.sensordata[self._feet_terrain_sensor_adrs] > 0, axis=-1
    )
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt

    # Track swing peak relative to deck surface so feet_height/clearance work correctly.
    p_fz = data.site_xpos[self._feet_site_id, -1] - _TERRAIN_Z
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    if self._config.history_len > 1:
      current_proprio = obs["state"][:_PROPRIO_SIZE]
      state.info["obs_history"] = jp.concatenate(
          [state.info["obs_history"][1:], current_proprio[None]], axis=0
      )
    failure = self._get_failure(data)
    success = self._get_success(data)
    done = failure | success

    # Track furthest x reached (metric only — reward uses velocity tracking).
    new_max = jp.maximum(state.info["max_x_reached"], data.qpos[0])
    state.info["max_x_reached"] = new_max

    rewards = self._get_reward(
        data, action, state.info, failure, success, first_contact, contact_filt
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.nan_to_num(
        jp.clip(sum(rewards.values()) * self.dt, -10.0, 10000.0), nan=0.0
    )

    # Snapshot air_time BEFORE zeroing — first_contact feet are in contact this
    # step, so their air_time would be wiped if we read after the reset below.
    air_time_at_td = (state.info["feet_air_time"] * first_contact).sum()

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_qvel"] = data.qvel
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    foot_y = data.site_xpos[self._feet_site_id, 1]
    state.metrics["metric/max_x_reached"] = state.info["max_x_reached"]
    state.metrics["metric/x_vel"] = self.get_local_linvel(data)[0]
    state.metrics["metric/touchdown_count"] = first_contact.sum().astype(jp.float32)
    state.metrics["metric/air_time_sum_at_td"] = air_time_at_td
    state.metrics["metric/term_fall"] = failure.astype(jp.float32)
    state.metrics["metric/term_success"] = success.astype(jp.float32)
    state.metrics["metric/abduction_saturation"] = jp.mean(
        (jp.abs(action[0::3]) > 0.95).astype(jp.float32)
    )
    state.metrics["metric/max_foot_y"] = jp.max(jp.abs(foot_y))
    state.metrics["metric/virtual_hw"] = state.info["virtual_hw"]
    state.metrics["metric/swing_peak"] = jp.mean(state.info["swing_peak"])

    # ── Episode-boundary resets (fix BraxAutoResetWrapper info leak) ──────────
    # `done` catches failure/success. `teleported` catches EpisodeWrapper timeouts
    # (AutoReset restores spawn data/obs but not info, so we must clean up both).
    boundary = done | teleported
    state.info["rng"], key = jax.random.split(state.info["rng"])
    vhw_min = min(self._config.virtual_hw_min, self._config.bridge_half_width)
    new_vhw = jax.random.uniform(
        key, (),
        minval=vhw_min,
        maxval=self._config.bridge_half_width,
    )
    state.info["virtual_hw"] = jp.where(boundary, new_vhw, state.info["virtual_hw"])
    state.info["max_x_reached"] = jp.where(
        boundary, state.info["init_x"], state.info["max_x_reached"]
    )
    state.info["feet_air_time"] = jp.where(
        boundary, jp.zeros(4), state.info["feet_air_time"]
    )
    state.info["last_contact"] = jp.where(
        boundary, jp.zeros(4, dtype=bool), state.info["last_contact"]
    )
    state.info["swing_peak"] = jp.where(
        boundary, jp.zeros(4), state.info["swing_peak"]
    )
    state.info["last_act"] = jp.where(
        boundary, jp.zeros_like(state.info["last_act"]), state.info["last_act"]
    )
    state.info["last_last_act"] = jp.where(
        boundary, jp.zeros_like(state.info["last_last_act"]), state.info["last_last_act"]
    )
    state.info["last_qvel"] = jp.where(
        boundary, jp.zeros_like(state.info["last_qvel"]), state.info["last_qvel"]
    )
    state.info["obs_history"] = jp.where(
        boundary, jp.zeros_like(state.info["obs_history"]), state.info["obs_history"]
    )

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _get_failure(self, data: mjx.Data) -> jax.Array:
    tilt = self.get_upvector(data)[-1] < 0.5  # terminate at ~60° tilt
    height = data.qpos[2] < self._config.fall_threshold
    nan_explosion = ~jp.isfinite(data.qpos).all()
    return tilt | height | nan_explosion

  def _get_success(self, data: mjx.Data) -> jax.Array:
    # Robot has confidently crossed — reached halfway into Platform B.
    return data.qpos[0] >= self._config.goal_x

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    # Progress along the 7 m terrain span, normalised to roughly [-1, 1].
    x_progress = jp.clip(data.qpos[0] / 7.0, -1.0, 1.0)

    local_heightmap = self._get_local_heightmap(data)  # (169,)

    # PUMA foothold prior: signed lateral y-distance of each foot from bridge
    # centreline (y=0). Positive = right of centre, negative = left. Tiny
    # Gaussian noise simulates real-world estimation error; defaults to 0.0.
    foot_y = data.site_xpos[self._feet_site_id, 1]  # (4,) FR, FL, RR, RL
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_foot_y = (
        foot_y
        + jax.random.normal(noise_rng, (4,))
        * self._config.noise_config.level
        * self._config.noise_config.scales.foot_y
    )

    proprio = jp.hstack([
        noisy_linvel,                             # 3
        noisy_gyro,                               # 3
        noisy_gravity,                            # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,                          # 12
        info["last_act"],                         # 12
        jp.array([x_progress]),                   # 1
        noisy_foot_y,                             # 4 — PUMA foothold prior
        jp.array([data.qpos[1]]),                 # 1 — trunk y (lateral drift)
        jp.array([info["virtual_hw"] / 0.4]),              # 1 — virtual bridge width normalised
    ])  # total: 52 = _PROPRIO_SIZE

    if self._config.history_len > 1:
      state = jp.hstack([
          proprio,                                # 52
          local_heightmap,                        # 169
          info["obs_history"].ravel(),            # (history_len-1) * 52
      ])
    else:
      state = jp.hstack([proprio, local_heightmap])  # 221

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,                                     # 3
        accelerometer,                            # 3
        gravity,                                  # 3
        linvel,                                   # 3
        angvel,                                   # 3
        joint_angles - self._default_pose,        # 12
        joint_vel,                                # 12
        data.actuator_force,                      # 12
        info["last_contact"],                     # 4
        feet_vel,                                 # 12
        info["feet_air_time"],                    # 4
        jp.array([data.qpos[0]]),                 # 1
    ])

    return {"state": state, "privileged_state": privileged_state}

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      failure: jax.Array,
      success: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    # Foothold quality restricted to contacting feet only.
    foot_y = data.site_xpos[self._feet_site_id, 1]
    margin = info["virtual_hw"] - jp.abs(foot_y)
    quality = jp.clip(margin / jp.minimum(info["virtual_hw"], 0.10), 0.0, 1.0)
    contact_f = contact.astype(jp.float32)
    n_contact = jp.maximum(contact_f.sum(), 1.0)
    quality_mean = jp.sum(quality * contact_f) / n_contact
    foothold_multiplier = 0.5 + 0.5 * quality_mean  # [0.5, 1.0]

    global_linvel = self.get_global_linvel(data)
    global_angvel = self.get_global_angvel(data)

    return {
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "alive": jp.ones(()),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "action_accel": self._cost_action_accel(
            action, info["last_act"], info["last_last_act"]
        ),
        "dof_acc": self._cost_dof_acc(data.qvel, info["last_qvel"]),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "termination": self._cost_termination(failure),
        "success": self._reward_success(success),
        "tracking_lin_vel": self._reward_tracking_lin_vel(data, foothold_multiplier),
        "lateral_deviation": self._cost_lateral_deviation(data.qpos[1]),
        "heading": self._cost_heading(
            data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact
        ),
        "foot_off_bridge": self._cost_foot_off_bridge(data, first_contact),
        "lin_vel_z": self._cost_lin_vel_z(global_linvel),
        "ang_vel_xy": self._cost_ang_vel_xy(global_angvel),
        "feet_slip": self._cost_feet_slip(data, contact),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(info["swing_peak"], first_contact),
        "pose": self._reward_pose(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }

  def _reward_forward_vel(self, local_vel: jax.Array) -> jax.Array:
    return jp.clip(local_vel[0], 0.0, 2.0)

  def _reward_progress_to_goal(self, max_x_reached: jax.Array) -> jax.Array:
    # Log-shaped gradient from spawn (x=-1.5) to goal (x=5.5).
    # Accepts max_x_reached (monotonically non-decreasing) so that backward
    # movement never improves this reward — eliminating the oscillation local
    # optimum where the robot averages a moderate x position by going back and
    # forth instead of pushing steadily forward.
    x_shifted = jp.clip(max_x_reached + 2.5, 1.0, 8.0)
    return jp.log(x_shifted) / jp.log(jp.array(8.0))

  def _cost_lateral_deviation(self, y_pos: jax.Array) -> jax.Array:
    return jp.square(y_pos)

  def _cost_heading(self, forward_vec: jax.Array) -> jax.Array:
    # 0 when facing +x, 2 when facing -x. Distinguishes forward from backward,
    # unlike the old square(y) which was symmetric about ±x.
    return 1.0 - jp.clip(forward_vec[0], -1.0, 1.0)

  def _cost_orientation(self, upvector: jax.Array) -> jax.Array:
    return jp.sum(jp.square(upvector[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))

  def _cost_action_accel(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.square(act - 2.0 * last_act + last_last_act))

  def _cost_dof_acc(self, qvel: jax.Array, last_qvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square((qvel - last_qvel) / self.dt))

  def _reward_tracking_lin_vel(
      self, data: mjx.Data, foothold_multiplier: jax.Array
  ) -> jax.Array:
    local_vel = self.get_local_linvel(data)
    target = jp.array([self._config.target_speed, 0.0])
    err = jp.sum(jp.square(target - local_vel[:2]))
    return jp.exp(-err / 0.25) * foothold_multiplier

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_success(self, success: jax.Array) -> jax.Array:
    return success.astype(jp.float32)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array
  ) -> jax.Array:
    # Cap at 0.25s: healthy trot swings earn positively; jitter <0.1s pays penalty.
    capped = jp.minimum(air_time, 0.25)
    return jp.sum((capped - 0.1) * first_contact)

  def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact.astype(jp.float32))

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    # Subtract deck z so clearance is relative to terrain surface, not world origin.
    foot_z = data.site_xpos[self._feet_site_id, 2] - _TERRAIN_Z
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self, swing_peak: jax.Array, first_contact: jax.Array
  ) -> jax.Array:
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_foot_off_bridge(
      self, data: mjx.Data, first_contact: jax.Array
  ) -> jax.Array:
    foot_xy = data.site_xpos[self._feet_site_id, :2]  # (4, 2)
    nx, ny = self._world_heightmap.shape
    xi = jp.clip(
        jp.round((foot_xy[:, 0] - _HM_X_MIN) / _HM_CELL).astype(jp.int32),
        0, nx - 1,
    )
    yi = jp.clip(
        jp.round((foot_xy[:, 1] - _HM_Y_MIN) / _HM_CELL).astype(jp.int32),
        0, ny - 1,
    )
    on_surface = self._world_heightmap[xi, yi]  # (4,) binary
    return jp.sum(first_contact * (1.0 - on_surface))

  def _cost_backward_vel(self, local_vel: jax.Array) -> jax.Array:
    """Penalise backward (-x) velocity to deter back-and-forth oscillation.

    Returns the magnitude of any backward velocity (≥ 0). Combined with the
    negative scale in reward_config, this imposes a cost symmetric to the
    forward_vel reward, eliminating the free ride that oscillation gets from
    forward strokes yielding reward while backward strokes cost nothing.
    """
    return jp.maximum(0.0, -local_vel[0])
