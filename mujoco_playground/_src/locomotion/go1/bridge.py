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

_PLATFORM_A_X_MIN = -3.0
_PLATFORM_A_X_MAX = 0.0
_BRIDGE_X_MIN = 0.0
_BRIDGE_X_MAX = 4.0
_PLATFORM_B_X_MIN = 4.0
_PLATFORM_B_X_MAX = 7.0
_PLATFORM_HALF_WIDTH = 1.0
_FOOT_RADIUS = 0.023

# linvel(3)+gyro(3)+gravity(3)+joint angles(12)+joint vel(12)+
# contacts(4)+goal direction(2)+virtual bridge half-width(1)+last action(12)
_PROPRIO_SIZE = 52


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=20.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.3,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      # Curriculum: bridge y half-extent in metres (0.4 = 0.8 m wide).
      bridge_half_width=0.4,
      # Per-episode virtual half-width used by the reward. The physical bridge
      # stays at bridge_half_width; virtual width creates within-stage pressure
      # to narrow stance before the next curriculum stage.
      virtual_half_width_min=0.15,
      frontier_progress_cap=0.6,
      # Terminate if root z drops below this (0.3 m below the 0.5 m platform surface).
      fall_threshold=0.2,
      # Terminate with success bonus when robot reaches this x (halfway into Platform B).
      # Platform B spans x=4.0 to x=7.0; midpoint is 5.5.
      goal_x=5.5,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              frontier_delta=50.0,
              feet_air_time=2.0,
              success=5000.0,
              pose=0.5,
              termination=-200.0,
              foot_off_bridge=-50.0,
              foot_off_virtual=-10.0,
              feet_stale_air=-2.0,
              orientation=-5.0,
              lateral_deviation=-3.0,
              heading=-2.0,
              lin_vel_z=-2.0,
              feet_clearance=-2.0,
              feet_slip=-0.25,
              dof_pos_limits=-1.0,
              action_rate=-0.01,
              torques=-0.0002,
              energy=-0.001,
              feet_height=-0.2,
          ),
          max_foot_height=0.1,
          feet_air_time_min=0.10,
          feet_air_time_cap=0.15,
          stale_air_time=0.4,
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
    n_half = int(round((_HM_Y_MAX - _HM_Y_MIN) / 2 / _HM_CELL))
    ys = np.arange(-n_half, n_half + 1) * _HM_CELL
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    hw = self._config.bridge_half_width
    on_platform_a = (
        (xx >= _PLATFORM_A_X_MIN)
        & (xx <= _PLATFORM_A_X_MAX)
        & (np.abs(yy) <= _PLATFORM_HALF_WIDTH)
    )
    on_bridge = (
        (xx >= _BRIDGE_X_MIN)
        & (xx <= _BRIDGE_X_MAX)
        & (np.abs(yy) <= hw)
    )
    on_platform_b = (
        (xx >= _PLATFORM_B_X_MIN)
        & (xx <= _PLATFORM_B_X_MAX)
        & (np.abs(yy) <= _PLATFORM_HALF_WIDTH)
    )
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

    rng, key = jax.random.split(rng)
    vhw_min = jp.minimum(
        jp.array(self._config.virtual_half_width_min),
        jp.array(self._config.bridge_half_width),
    )
    virtual_hw = jax.random.uniform(
        key,
        (),
        minval=vhw_min,
        maxval=jp.array(self._config.bridge_half_width),
    )

    info = {
        "rng": rng,
        "virtual_hw": virtual_hw,
        "max_x_reached": qpos[0],
        "unpaid_frontier": jp.zeros(()),
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "obs_history": jp.zeros((self._config.history_len - 1, _PROPRIO_SIZE)),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    for k in (
        "metric/term_success",
        "metric/term_failure",
        "metric/touchdown_count",
        "metric/contact_left",
        "metric/contact_right",
        "metric/max_foot_y",
        "metric/foothold_multiplier",
        "metric/virtual_hw",
        "metric/abduction_saturation",
    ):
      metrics[k] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    # True if any terrain surface is under that foot.
    contact = jp.any(
        data.sensordata[self._feet_terrain_sensor_adrs] > 0, axis=-1
    )
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt

    prev_max_x = state.info["max_x_reached"]
    max_x_reached = jp.maximum(prev_max_x, data.qpos[0])
    frontier_delta = jp.minimum(
        jp.maximum(max_x_reached - prev_max_x, 0.0),
        self._config.frontier_progress_cap * self.dt,
    )
    state.info["max_x_reached"] = max_x_reached
    state.info["unpaid_frontier"] += frontier_delta

    p_fz = data.site_xpos[self._feet_site_id, -1]
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

    rewards = self._get_reward(
        data, action, state.info, failure, success, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.nan_to_num(
        jp.clip(sum(rewards.values()) * self.dt, -10.0, 10000.0), nan=0.0
    )

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["unpaid_frontier"] = jp.where(
        jp.any(first_contact), 0.0, state.info["unpaid_frontier"]
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    self._update_metrics(state.metrics, data, state.info, success, failure,
                         first_contact, contact)

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _get_failure(self, data: mjx.Data) -> jax.Array:
    tilt = self.get_upvector(data)[-1] < jp.cos(0.7)
    height = data.qpos[2] < self._config.fall_threshold
    return tilt | height

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

    contact = info["last_contact"].astype(jp.float32)
    goal_dir = self._goal_direction_local(data)
    virtual_hw = jp.array([info["virtual_hw"] / 0.4])

    proprio = jp.hstack([
        noisy_linvel,                             # 3
        noisy_gyro,                               # 3
        noisy_gravity,                            # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,                          # 12
        contact,                                  # 4
        goal_dir,                                 # 2
        virtual_hw,                               # 1
        info["last_act"],                         # 12
    ])  # total: 52 = _PROPRIO_SIZE

    if self._config.history_len > 1:
      state = jp.hstack([
          proprio,
          info["obs_history"].ravel(),
      ])
    else:
      state = proprio

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
        jp.array([info["virtual_hw"]]),           # 1
        jp.array([info["max_x_reached"]]),        # 1
        jp.array([info["unpaid_frontier"]]),      # 1
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
    return {
        "frontier_delta": self._reward_frontier_delta(data, info, first_contact),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "termination": self._cost_termination(failure),
        "success": self._reward_success(success),
        "pose": self._reward_pose(data.qpos[7:]),
        "lateral_deviation": self._cost_lateral_deviation(data.qpos[1]),
        "heading": self._cost_heading(
            data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact
        ),
        "foot_off_bridge": self._cost_foot_off_bridge(data, first_contact),
        "foot_off_virtual": self._cost_foot_off_virtual(data, info),
        "feet_stale_air": self._cost_feet_stale_air(info["feet_air_time"]),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_slip": self._cost_feet_slip(data, contact),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact
        ),
    }

  def _reward_frontier_delta(
      self, data: mjx.Data, info: dict[str, Any], first_contact: jax.Array
  ) -> jax.Array:
    payout = info["unpaid_frontier"] / self.dt
    return payout * self._foothold_multiplier(data, info) * jp.any(first_contact)

  def _cost_lateral_deviation(self, y_pos: jax.Array) -> jax.Array:
    return jp.square(y_pos)

  def _cost_heading(self, forward_vec: jax.Array) -> jax.Array:
    # Penalise the y-component of the robot's forward vector.
    # Zero when facing +x, maximum (1.0) when facing sideways (±y).
    return jp.square(forward_vec[1])

  def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
    return jp.square(global_linvel[2])

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

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_success(self, success: jax.Array) -> jax.Array:
    return success.astype(jp.float32)

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    weight = jp.array([0.1, 1.0, 1.0] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array
  ) -> jax.Array:
    capped_air = jp.minimum(
        air_time, self._config.reward_config.feet_air_time_cap
    )
    return jp.sum(
        jp.maximum(capped_air - self._config.reward_config.feet_air_time_min, 0.0)
        * first_contact
    )

  def _cost_foot_off_bridge(
      self, data: mjx.Data, first_contact: jax.Array
  ) -> jax.Array:
    foot_xy = data.site_xpos[self._feet_site_id, :2]  # (4, 2)
    on_surface = self._on_surface(foot_xy).astype(jp.float32)
    return jp.sum(first_contact * (1.0 - on_surface))

  def _cost_foot_off_virtual(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    foot_y = data.site_xpos[self._feet_site_id, 1]  # (4,) world-frame y
    return jp.sum(jp.maximum(jp.abs(foot_y) + _FOOT_RADIUS - info["virtual_hw"], 0.0))

  def _cost_feet_stale_air(self, air_time: jax.Array) -> jax.Array:
    return jp.sum(jp.maximum(air_time - self._config.reward_config.stale_air_time, 0.0))

  def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    return jp.sum(jp.sum(jp.square(vel_xy), axis=-1) * contact)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_z = data.site_xpos[self._feet_site_id, -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self, swing_peak: jax.Array, first_contact: jax.Array
  ) -> jax.Array:
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _foothold_quality(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    foot_y = data.site_xpos[self._feet_site_id, 1]
    tau = jp.maximum(0.5 * info["virtual_hw"], 0.01)
    return jp.exp(jp.minimum(info["virtual_hw"] - jp.abs(foot_y), 0.0) / tau)

  def _foothold_multiplier(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    quality_mean = jp.mean(self._foothold_quality(data, info))
    floor = 0.5 * jp.clip((info["virtual_hw"] - 0.10) / 0.05, 0.0, 1.0)
    return floor + (1.0 - floor) * quality_mean

  def _on_surface(self, xy: jax.Array) -> jax.Array:
    x = xy[..., 0]
    y = xy[..., 1]
    on_platform_a = (
        (x >= _PLATFORM_A_X_MIN)
        & (x <= _PLATFORM_A_X_MAX)
        & (jp.abs(y) <= _PLATFORM_HALF_WIDTH)
    )
    on_bridge = (
        (x >= _BRIDGE_X_MIN)
        & (x <= _BRIDGE_X_MAX)
        & (jp.abs(y) <= self._config.bridge_half_width)
    )
    on_platform_b = (
        (x >= _PLATFORM_B_X_MIN)
        & (x <= _PLATFORM_B_X_MAX)
        & (jp.abs(y) <= _PLATFORM_HALF_WIDTH)
    )
    return on_platform_a | on_bridge | on_platform_b

  def _goal_direction_local(self, data: mjx.Data) -> jax.Array:
    goal_world = jp.array([self._config.goal_x - data.qpos[0], -data.qpos[1]])
    norm = jp.linalg.norm(goal_world) + 1e-6
    goal_world = goal_world / norm

    forward = data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
    fwd_norm = jp.linalg.norm(forward[:2]) + 1e-6
    cos_yaw = forward[0] / fwd_norm
    sin_yaw = forward[1] / fwd_norm
    left = jp.array([-sin_yaw, cos_yaw])
    fwd = jp.array([cos_yaw, sin_yaw])
    return jp.array([jp.dot(goal_world, fwd), jp.dot(goal_world, left)])

  def _abduction_saturation(self, qpos: jax.Array) -> jax.Array:
    abduction = qpos[0::3]
    lowers = self._soft_lowers[0::3]
    uppers = self._soft_uppers[0::3]
    margin = 0.05
    saturated = (abduction < lowers + margin) | (abduction > uppers - margin)
    return jp.mean(saturated.astype(jp.float32))

  def _update_metrics(
      self,
      metrics: dict[str, Any],
      data: mjx.Data,
      info: dict[str, Any],
      success: jax.Array,
      failure: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> None:
    foot_y = data.site_xpos[self._feet_site_id, 1]
    metrics["metric/term_success"] = success.astype(jp.float32)
    metrics["metric/term_failure"] = (failure & ~success).astype(jp.float32)
    metrics["metric/touchdown_count"] = jp.sum(
        first_contact.astype(jp.float32)
    )
    metrics["metric/contact_left"] = jp.sum(
        contact[jp.array([1, 3])].astype(jp.float32)
    )
    metrics["metric/contact_right"] = jp.sum(
        contact[jp.array([0, 2])].astype(jp.float32)
    )
    metrics["metric/max_foot_y"] = jp.max(jp.abs(foot_y))
    metrics["metric/foothold_multiplier"] = self._foothold_multiplier(data, info)
    metrics["metric/virtual_hw"] = info["virtual_hw"]
    metrics["metric/abduction_saturation"] = self._abduction_saturation(
        data.qpos[7:]
    )
