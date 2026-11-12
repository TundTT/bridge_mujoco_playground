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
"""Bridge crossing for Go1 with an actuated roll tail and random pushes.

The tail (see xmls/go1_mjx_feetonly_tail.xml) is a 0.5 m rod with a 1.134 kg
tip mass driven by a modelled CubeMars AK70-10 about the trunk x axis, so it
produces roll. It is actuator index 12; the policy outputs 13 actions.

tail_mode:
  "active": the policy commands the tail target in [-90, +90] deg.
  "locked": the tail action is ignored and the motor holds the tail upright.
            Same mass/inertia as "active" -> the control condition for the
            "does the tail help under pushes?" comparison.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import bridge
from mujoco_playground._src.locomotion.go1 import go1_constants as consts

_TAIL_JOINT = "tail_joint"
_TAIL_ACTUATOR = "tail"


def default_config() -> config_dict.ConfigDict:
  config = bridge.default_config()
  with config.unlocked():
    config.tail_mode = "active"
    # Policy action in [-1, 1] maps to a tail target of +/- tail_action_scale.
    # The XML ctrlrange also clamps the target to +/- 90 deg.
    config.tail_action_scale = float(np.pi / 2)
    # AK70-10 servo gains. Go1Env overwrites every actuator with the leg Kp/Kd,
    # so these are re-applied to the tail after the model is loaded.
    config.tail_kp = 100.0
    config.tail_kv = 5.0  # AK70-10 MIT-mode Kd maximum (driver manual v1.0.18)
    config.tail_damping = 0.0
    # Safe envelope: past this the rod is close to the trunk (contact at 120 deg).
    config.tail_safe_angle = float(np.pi / 2)
    # Terminate (as a failure) if the tail gets this close to hitting the body.
    config.tail_terminate_angle = float(np.deg2rad(110.0))
    # AK70-10 rated (continuous) torque and the time constant of the running
    # mean-square torque used as a thermal proxy.
    config.tail_rated_torque = 8.3
    config.tail_thermal_tau = 5.0

    config.reset_info_on_new_episode = True
    config.push_config.enable = True

    config.reward_config.scales.tail_over_limit = -20.0
    config.reward_config.scales.tail_overtravel = -500.0
    config.reward_config.scales.tail_thermal = -1.0
    config.reward_config.scales.tail_energy = -0.001
    # Level-walking tail terms (off by default; see level-walking notes).
    # tail_center: tail angle^2 (rad^2) -- discourages parking the tail to one
    #   side as a static counterweight.
    # tail_roll_assist: positive power the tail's reaction torque removes from
    #   trunk roll, max(tau * omega_x, 0) capped at tail_assist_cap (W).
    config.reward_config.scales.tail_center = 0.0
    config.reward_config.scales.tail_roll_assist = 0.0
    config.tail_assist_cap = 10.0
  return config


class BridgeCrossingTail(bridge.BridgeCrossing):
  """Cross the narrow bridge with an actuated roll tail under random pushes."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=consts.FEET_ONLY_BRIDGE_TAIL_XML.as_posix(),
    )
    if self._config.tail_mode not in ("active", "locked"):
      raise ValueError(f"Unsupported tail_mode: {self._config.tail_mode}")

  def _customize_mj_model(self) -> None:
    m = self._mj_model
    act = m.actuator(_TAIL_ACTUATOR).id
    dof = m.jnt_dofadr[m.joint(_TAIL_JOINT).id]
    m.actuator_gainprm[act, 0] = self._config.tail_kp
    m.actuator_biasprm[act, 1] = -self._config.tail_kp
    m.actuator_biasprm[act, 2] = -self._config.tail_kv
    m.dof_damping[dof] = self._config.tail_damping

  def _post_init(self) -> None:
    super()._post_init()
    m = self._mj_model
    self._tail_act = m.actuator(_TAIL_ACTUATOR).id
    self._tail_qadr = int(m.jnt_qposadr[m.joint(_TAIL_JOINT).id])
    self._tail_vadr = int(m.jnt_dofadr[m.joint(_TAIL_JOINT).id])
    assert self._tail_act == 12 and self._tail_qadr == 19, "tail must follow legs"
    self._action_scale = self._action_scale.at[self._tail_act].set(
        self._config.tail_action_scale
    )

  # ── Reset / step ──────────────────────────────────────────────────────────

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)
    state.info["tail_tau_sq_ema"] = jp.zeros(())
    for k in ("tail_angle_abs", "tail_over_limit", "tail_torque_abs",
              "tail_torque_rms", "term_tail"):
      state.metrics[f"metric/{k}"] = jp.zeros(())
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.tail_mode == "locked":
      action = action.at[self._tail_act].set(0.0)
    state = super().step(state, action)

    data = state.data
    q = data.qpos[self._tail_qadr]
    tau = data.actuator_force[self._tail_act]
    # Thermal proxy (lagged one step for the reward): EMA of tau^2.
    alpha = self.dt / self._config.tail_thermal_tau
    ema = (1.0 - alpha) * state.info["tail_tau_sq_ema"] + alpha * tau**2
    state.info["tail_tau_sq_ema"] = jp.where(state.done > 0, 0.0, ema)

    state.metrics["metric/tail_angle_abs"] = jp.rad2deg(jp.abs(q))
    state.metrics["metric/tail_over_limit"] = (
        jp.abs(q) > self._config.tail_safe_angle
    ).astype(jp.float32)
    state.metrics["metric/tail_torque_abs"] = jp.abs(tau)
    state.metrics["metric/tail_torque_rms"] = jp.sqrt(ema)
    state.metrics["metric/term_tail"] = self._tail_failure(data).astype(
        jp.float32
    )
    return state

  def _reset_episode_info(
      self, info: dict[str, Any], boundary: jax.Array
  ) -> None:
    super()._reset_episode_info(info, boundary)
    if "tail_tau_sq_ema" in info:
      info["tail_tau_sq_ema"] = jp.where(
          boundary, 0.0, info["tail_tau_sq_ema"]
      )

  # ── Termination / reward ──────────────────────────────────────────────────

  def _tail_failure(self, data: mjx.Data) -> jax.Array:
    return jp.abs(data.qpos[self._tail_qadr]) > self._config.tail_terminate_angle

  def _get_failure(self, data: mjx.Data) -> jax.Array:
    return super()._get_failure(data) | self._tail_failure(data)

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      failure: jax.Array,
      success: jax.Array,
      first_contact: jax.Array,
      contact_filt: jax.Array,
  ) -> dict[str, jax.Array]:
    rewards = super()._get_reward(
        data, action, info, failure, success, first_contact, contact_filt
    )
    # Keep the leg torque/energy terms identical to the tail-less env; the
    # tail has its own terms below.
    leg_force = data.actuator_force[:12]
    rewards["torques"] = self._cost_torques(leg_force)
    rewards["energy"] = self._cost_energy(data.qvel[6:18], leg_force)

    q = data.qpos[self._tail_qadr]
    qd = data.qvel[self._tail_vadr]
    tau = data.actuator_force[self._tail_act]
    over = jp.maximum(jp.abs(q) - self._config.tail_safe_angle, 0.0)
    rewards["tail_over_limit"] = (over > 0.0).astype(jp.float32)
    rewards["tail_overtravel"] = over
    rewards["tail_thermal"] = jp.maximum(
        jp.sqrt(info["tail_tau_sq_ema"]) - self._config.tail_rated_torque, 0.0
    )
    rewards["tail_energy"] = jp.abs(qd * tau)
    rewards["tail_center"] = jp.square(q)
    # The motor applies tau to the tail and -tau to the trunk about the trunk
    # x axis, so the tail opposes the trunk's roll rate when tau * omega_x > 0.
    roll_rate = self.get_gyro(data)[0]
    rewards["tail_roll_assist"] = jp.clip(
        tau * roll_rate, 0.0, self._config.tail_assist_cap
    )
    return rewards
