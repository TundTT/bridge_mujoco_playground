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


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.3,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      # Curriculum: bridge y half-extent in metres (0.4 = 0.8 m wide).
      bridge_half_width=0.4,
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
              forward_vel=2.0,
              orientation=-5.0,
              alive=0.1,
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              termination=-1.0,
              success=300.0,
              feet_air_time=0.1,
              progress_to_goal=3.0,
              lateral_deviation=-2.0,
              heading=-2.0,
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

    self._post_init()

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

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

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

    p_fz = data.site_xpos[self._feet_site_id, -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
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
        jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0), nan=0.0
    )

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _get_failure(self, data: mjx.Data) -> jax.Array:
    tilt = self.get_upvector(data)[-1] < 0.0
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

    # Progress along the 7 m terrain span, normalised to roughly [-1, 1].
    x_progress = jp.clip(data.qpos[0] / 7.0, -1.0, 1.0)

    state = jp.hstack([
        noisy_linvel,                             # 3
        noisy_gyro,                               # 3
        noisy_gravity,                            # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,                          # 12
        info["last_act"],                         # 12
        jp.array([x_progress]),                   # 1
    ])

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
    return {
        "forward_vel": self._reward_forward_vel(self.get_local_linvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "alive": jp.ones(()),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "termination": self._cost_termination(failure),
        "success": self._reward_success(success),
        "progress_to_goal": self._reward_progress_to_goal(data.qpos[0]),
        "lateral_deviation": self._cost_lateral_deviation(data.qpos[1]),
        "heading": self._cost_heading(
            data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact
        ),
    }

  def _reward_forward_vel(self, local_vel: jax.Array) -> jax.Array:
    return jp.clip(local_vel[0], 0.0, 2.0)

  def _reward_progress_to_goal(self, x_pos: jax.Array) -> jax.Array:
    # Log-shaped gradient from spawn (x=-1.5) to goal (x=5.5).
    # Steep near spawn for fast early learning, tapers toward goal so the
    # success bonus dominates the final push onto Platform B.
    # Shift so log argument = 1.0 at spawn (log=0) and 8.0 at goal (log=2.079).
    x_shifted = jp.clip(x_pos + 2.5, 1.0, 8.0)
    return jp.log(x_shifted) / jp.log(jp.array(8.0))

  def _cost_lateral_deviation(self, y_pos: jax.Array) -> jax.Array:
    return jp.square(y_pos)

  def _cost_heading(self, forward_vec: jax.Array) -> jax.Array:
    # Penalise the y-component of the robot's forward vector.
    # Zero when facing +x, maximum (1.0) when facing sideways (±y).
    return jp.square(forward_vec[1])

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

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array
  ) -> jax.Array:
    return jp.sum((air_time - 0.1) * first_contact)
