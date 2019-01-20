# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A class implementing minimal Atari 2600 preprocessing.

This includes:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces.box import Box
import gym
import numpy as np
import gin.tf
#import cv2


@gin.configurable
class GymPreprocessing(object):
  """A class implementing image preprocessing for Gym agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  在由于不处理图像问题，所以暂时frame_skip设置为1，screen_size设置为0，目的是拿掉所有sreen_size可能影响的人
  """

  def __init__(self, frame_skip=1, terminal_on_life_loss=False,
               screen_size=0):
    self.environment = gym.make('Pendulum-v0')
    self.environment.reset()

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return self.environment.observation_space

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return Box(0,1,dtype=np.float32)

  @property
  def metadata(self):
    return self.environment.metadata

  def reset(self):

    return self.environment.reset()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    observation, reward, game_over, info = self.environment.step(action)
    self.game_over = game_over
    return observation, reward, game_over, info

