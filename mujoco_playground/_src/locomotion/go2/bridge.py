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
"""Bridge crossing (with and without the roll tail) for the Unitree Go2.

Same task, rewards, observations and curriculum as the Go1 envs in
go1/bridge.py and go1/bridge_tail.py. Go2-specific settings:
  - Kp 50 (Go2 menagerie gain). At the Go1's Kp 35 the heavier Go2 (15.2 kg,
    17.1 kg with the tail) sags 8-22 deg nose-up standing still; at Kp 50 it
    stands like the Go1 does at Kp 35.
  - foot_radius 0.0175 (Go2 foot sphere).
Fixes that were opt-in for the Go1 are on by default here:
  - foot_height_from_surface: swing-height rewards measure from the 0.5 m deck.
  - reset_info_on_new_episode: per-episode info is reset after a timeout.
Added on request: a pitch penalty with a +/-10 deg dead band, so the trunk
stays roughly level but can still pitch a little for balance.
"""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from mujoco_playground._src.locomotion.go1 import bridge
from mujoco_playground._src.locomotion.go1 import bridge_tail
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


def _go2_settings(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  with config.unlocked():
    config.Kp = 50.0
    config.foot_radius = consts.FOOT_RADIUS
    config.foot_height_from_surface = True
    config.reset_info_on_new_episode = True
    config.pitch_deadband_deg = 10.0
    config.reward_config.scales.pitch = -75.0
  return config


def default_config() -> config_dict.ConfigDict:
  return _go2_settings(bridge.default_config())


def tail_default_config() -> config_dict.ConfigDict:
  return _go2_settings(bridge_tail.default_config())


class Go2BridgeCrossing(bridge.BridgeCrossing):
  """Go2 crosses the narrow bridge."""

  _root_body_name = consts.ROOT_BODY
  _default_xml = consts.FEET_ONLY_BRIDGE_XML

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    super().__init__(config, config_overrides, xml_path)

  def _load_assets(self) -> Dict[str, bytes]:
    return go2_base.get_assets()


class Go2BridgeCrossingTail(bridge_tail.BridgeCrossingTail):
  """Go2 with the AK70-10 roll tail crosses the narrow bridge."""

  _root_body_name = consts.ROOT_BODY
  _default_tail_xml = consts.FEET_ONLY_BRIDGE_TAIL_XML

  def __init__(
      self,
      config: config_dict.ConfigDict = tail_default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    super().__init__(config, config_overrides, xml_path)

  def _load_assets(self) -> Dict[str, bytes]:
    return go2_base.get_assets()
