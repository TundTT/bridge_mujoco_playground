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
"""Unitree Go2 constants for the bridge tasks."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go2"
FEET_ONLY_BRIDGE_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_bridge.xml"
FEET_ONLY_BRIDGE_TAIL_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_bridge_tail.xml"
)

# Sensor, site and foot-geom names are the same as the Go1 model's (see
# go1_constants); only the root body differs.
ROOT_BODY = "base"
FOOT_RADIUS = 0.0175
