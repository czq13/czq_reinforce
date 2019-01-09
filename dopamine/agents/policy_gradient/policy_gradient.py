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
"""Compact implementation of a DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random

from dopamine.replay_memory import policy_search_reply
import numpy as np
import tensorflow as tf

import gin.tf

slim = tf.contrib.slim

NATURE_GP_OBSERVATION_SHAPE = (4, 1)  # Size of downscaled Atari 2600 frame.
NATURE_GP_DTYPE = tf.float32  # DType of Atari 2600 observations.
NATURE_GP_STACK_SIZE = 1  # Number of frames in the state stack.

@gin.configurable
class PGAgent(object):
    """An implementation of the DQN agent."""

    def __init__(self,
                 sess,
                 num_actions,
                 observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=NATURE_DQN_DTYPE,
                 stack_size=NATURE_DQN_STACK_SIZE,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=None,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 tf_device='/cpu:*',
                 use_staging=True,
                 max_tf_checkpoints_to_keep=3,
                 optimizer=tf.train.RMSPropOptimizer(
                     learning_rate=0.00025,
                     decay=0.95,
                     momentum=0.0,
                     epsilon=0.00001,
                     centered=True),
                 summary_writer=None,
                 summary_writing_frequency=5):
        """Initializes the agent and constructs the components of its graph.

        Args:
          sess: `tf.Session`, for executing ops.
          num_actions: int, number of actions the agent can take at any state.
          observation_shape: tuple of ints describing the observation shape.
          observation_dtype: tf.DType, specifies the type of the observations. Note
            that if your inputs are continuous, you should set this to tf.float32.
          stack_size: int, number of frames to use in state stack.
          gamma: float, discount factor with the usual RL meaning.
          update_horizon: int, horizon at which updates are performed, the 'n' in
            n-step update.
          min_replay_history: int, number of transitions that should be experienced
            before the agent begins training its value function.
          update_period: int, period between DQN updates.
          target_update_period: int, update period for the target network.
          epsilon_fn: function expecting 4 parameters:
            (decay_period, step, warmup_steps, epsilon). This function should return
            the epsilon value used for exploration during training.
          epsilon_train: float, the value to which the agent's epsilon is eventually
            decayed during training.
          epsilon_eval: float, epsilon used when evaluating the agent.
          epsilon_decay_period: int, length of the epsilon decay schedule.
          tf_device: str, Tensorflow device on which the agent's graph is executed.
          use_staging: bool, when True use a staging area to prefetch the next
            training batch, speeding training up by about 30%.
          max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
            keep.
          optimizer: `tf.train.Optimizer`, for training the value function.
          summary_writer: SummaryWriter object for outputting training statistics.
            Summary writing disabled if set to None.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
        """
        assert isinstance(observation_shape, tuple)
        tf.logging.info('Creating %s agent with the following parameters:',
                        self.__class__.__name__)
        tf.logging.info('\t gamma: %f', gamma)
        tf.logging.info('\t update_horizon: %f', update_horizon)
        tf.logging.info('\t min_replay_history: %d', min_replay_history)
        tf.logging.info('\t update_period: %d', update_period)
        tf.logging.info('\t target_update_period: %d', target_update_period)
        tf.logging.info('\t epsilon_train: %f', epsilon_train)
        tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
        tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
        tf.logging.info('\t tf_device: %s', tf_device)
        tf.logging.info('\t use_staging: %s', use_staging)
        tf.logging.info('\t optimizer: %s', optimizer)

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.epsilon_fn = epsilon_fn
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.update_period = update_period
        self.eval_mode = False
        self.training_steps = 0
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.summary_writing_frequency = summary_writing_frequency

        with tf.device(tf_device):
            # Create a placeholder for the state input to the DQN network.
            # The last axis indicates the number of consecutive frames stacked.
            # state_shape=(1,4,1)
            state_shape = (1,) + self.observation_shape + (stack_size,)
            self.state = np.zeros(state_shape)
            # state_ph.shape=(1,4,1)
            self.state_ph = tf.placeholder(self.observation_dtype, state_shape,
                                           name='state_ph')
            self._replay = self._build_replay_buffer(use_staging)

            self._build_networks()

            self._train_op = self._build_train_op()
            self._sync_qt_ops = self._build_sync_op()

        if self.summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self._merged_summaries = tf.summary.merge_all()
        self._sess = sess
        self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)

        # Variables to be initialized by the agent once it interacts with the
        # environment.
        self._observation = None
        self._last_observation = None

    def _get_network_type(self):
        """Returns the type of the outputs of a Q value network.

        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple('PG_network', ['p_value'])

    def _network_template_for_atari(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

        Args:
          state: `tf.Tensor`, contains the agent's current state.

        Returns:
          net: _network_type object containing the tensors output by the network.
        """
        net = tf.cast(state, tf.float32)
        net = tf.div(net, 255.)
        net = slim.conv2d(net, 32, [8, 8], stride=4)
        net = slim.conv2d(net, 64, [4, 4], stride=2)
        net = slim.conv2d(net, 64, [3, 3], stride=1)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 512)
        q_values = slim.fully_connected(net, self.num_actions, activation_fn=None)
        return self._get_network_type()(q_values)

    def _network_template_no_dueling(self, state):
        cons = tf.constant([2 * 2.4, 100.0, 12 * 2 * 3.1415 / 360 * 2, 100.0])
        net = tf.cast(state, tf.float32)
        net = tf.div(net, cons)
        net = slim.fully_connected(net, 32)
        tnet = slim.fully_connected(net, 32)
        net = slim.flatten(tnet)
        q_values = slim.fully_connected(net, self.num_actions, activation_fn=None)
        return self._get_network_type()(q_values, tnet)

    def _network_template_for_dueling(self, state):
        cons = tf.constant([2 * 2.4, 100.0, 12 * 2 * 3.1415 / 360 * 2, 100.0])
        net = tf.cast(state, tf.float32)
        net = tf.div(net, cons)
        net = slim.fully_connected(net, 32)
        tnet = slim.fully_connected(net, 32)
        net = slim.flatten(tnet)
        # batch_size * (action_num + 1)
        net = slim.fully_connected(net, self.num_actions + 1, activation_fn=None)
        print(net)
        # value = tf.strided_slice(net,[0,0],[2,1],[1,1])
        value = net[:, :1]
        print(value.shape.as_list())
        # advantage = tf.strided_slice(net,[0,1],[2,4],[1,1])
        advantage = net[:, 1:]
        print(advantage.shape.as_list())
        q_values = value + advantage - tf.reduce_max(advantage, axis=1)[:, None]
        print(q_values.shape.as_list())
        return self._get_network_type()(q_values, tnet)

    def _network_template(self,state):
        cons = tf.constant([2 * 2.4, 100.0, 12 * 2 * 3.1415 / 360 * 2, 100.0])
        net = tf.cast(state,tf.float32)
        bnet = tf.div(net,cons)
        net = slim.fully_connected(bnet,32)
        net = slim.fully_connected(net,32)
        net = slim.fully_connected(net, self.num_actions, activation_fn=None)
        p_output = tf.contrib.layer.softmax(net)

        return self._get_network_type()(p_output)

    def _network_baseline(self,state):
        pass
    def _build_networks(self):
        """Builds the Q-value network computations needed for acting and training.

        These are:
          self.online_convnet: For computing the current state's Q-values.
          self.target_convnet: For computing the next state's target Q-values.
          self._net_outputs: The actual Q-values.
          self._q_argmax: The action maximizing the current state's Q-values.
          self._replay_net_outputs: The replayed states' Q-values.
          self._replay_next_target_net_outputs: The replayed next states' target
            Q-values (see Mnih et al., 2015 for details).
        """
        # Calling online_convnet will generate a new graph as defined in
        # self._get_network_template using whatever input is passed, but will always
        # share the same weights.
        self.online_convnet = tf.make_template('Online', self._network_template)
        self.target_convnet = tf.make_template('Target', self._network_template)
        # print('state_ph={}'.format(self.state_ph.shape.as_list()))
        self._net_outputs = self.online_convnet(self.state_ph)
        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        # using a deep network, but may affect performance with a linear
        # approximation scheme.
        # batch_size * action_nums

        self._replay_net_p_outputs = self.target_convnet(self._replay.states)
        self._replay_action_p = tf.gather_nd(self._replay_net_p_outputs,
                                             tf.concat([self._replay.indices,self._replay.actions],axis=1))
        self.loss = tf.reduce_mean(tf.multiply(tf.log(self._replay_action_p),self._replay.Gt-self._replay.baseline))




    def _build_replay_buffer(self, use_staging):
        """Creates the replay buffer used by the agent.

        Args:
          use_staging: bool, if True, uses a staging area to prefetch data for
            faster training.

        Returns:
          A WrapperReplayBuffer object.
        """
        return policy_search_reply.WrappedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype)

    def _build_target_q_op(self):
        """Build an op used as a target for the Q-value.

        Returns:
          target_q_op: An op calculating the Q-value.
        """
        # Get the maximum Q-value across the actions dimension.
        # batch_size
        replay_action = tf.argmax(self._replay_next_online_net_outputs.q_values, axis=1)[:, None]
        indices = tf.range(32, dtype=tf.int64)[:, None]
        rank = tf.concat([indices, replay_action], 1)
        print('our replay_action.shape={}'.format(replay_action.shape.as_list()))
        # replay_next_qt_max = tf.reduce_max(
        #    self._replay_next_target_net_outputs.q_values, 1)
        replay_next_qt_max = tf.gather_nd(self._replay_next_target_net_outputs.q_values, rank)
        # Calculate the Bellman target value.
        #   Q_t = R_t + \gamma^N * Q'_t+1
        # where,
        #   Q'_t+1 = \argmax_a Q(S_t+1, a)
        #          (or) 0 if S_t is a terminal state,
        # and
        #   N is the update horizon (by default, N=1).
        return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
                1. - tf.cast(self._replay.terminals, tf.float32))

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        replay_action_one_hot = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
        replay_chosen_q = tf.reduce_sum(
            self._replay_net_outputs.q_values * replay_action_one_hot,
            reduction_indices=1,
            name='replay_chosen_q')

        target = tf.stop_gradient(self._build_target_q_op())
        loss = tf.losses.huber_loss(
            target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))
        return self.optimizer.minimize(tf.reduce_mean(loss))

    def _build_sync_op(self):
        """Builds ops for assigning weights from online to target network.

        Returns:
          ops: A list of ops assigning weights from online to target network.
        """
        # Get trainable variables from online and target DQNs
        sync_qt_ops = []
        trainables_online = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online')
        trainables_target = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
        for (w_online, w_target) in zip(trainables_online, trainables_target):
            # Assign weights from online to target network.
            sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
        return sync_qt_ops

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """
        self._reset_state()
        self._record_observation(observation)

        if not self.eval_mode:
            self._train_step()

        self.action = self._select_action()
        return self.action

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.

        Returns:
          int, the selected action.
        """
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward, False)
            self._train_step()

        self.action = self._select_action()
        return self.action

    def end_episode(self, reward):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
          reward: float, the last reward from the environment.
        """
        if not self.eval_mode:
            self._store_transition(self._observation, self.action, reward, True)

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
           int, the selected action.
        """
        epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
            self.epsilon_decay_period,
            self.training_steps,
            self.min_replay_history,
            self.epsilon_train)
        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            return random.randint(0, self.num_actions - 1)
        else:
            # Choose the action with highest Q-value at the current state.
            return self._sess.run(self._q_argmax, {self.state_ph: self.state})

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.memory.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                self._sess.run(self._train_op)
                if (self.summary_writer is not None and
                        self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(summary, self.training_steps)

            if self.training_steps % self.target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

        self.training_steps += 1

    def _record_observation(self, observation):
        """Records an observation and update state.

        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.

        Args:
          observation: numpy array, an observation from the environment.
        """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        observation = np.reshape(observation, self.observation_shape)
        self._observation = observation[..., 0]
        self._observation = np.reshape(observation, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[0, ..., -1] = self._observation

    def _store_transition(self, last_observation, action, reward, is_terminal):
        """Stores an experienced transition.

        Executes a tf session and executes replay buffer ops in order to store the
        following tuple in the replay buffer:
          (last_observation, action, reward, is_terminal).

        Pedantically speaking, this does not actually store an entire transition
        since the next state is recorded on the following time step.

        Args:
          last_observation: numpy array, last observation.
          action: int, the action taken.
          reward: float, the reward.
          is_terminal: bool, indicating if the current state is a terminal state.
        """
        self._replay.add(last_observation, action, reward, is_terminal)

    def _reset_state(self):
        """Resets the agent state by filling it with zeros."""
        self.state.fill(0)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Args:
          checkpoint_dir: str, directory where TensorFlow objects will be saved.
          iteration_number: int, iteration number to use for naming the checkpoint
            file.

        Returns:
          A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        # Checkpoint the out-of-graph replay buffer.
        self._replay.save(checkpoint_dir, iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['state'] = self.state
        bundle_dictionary['eval_mode'] = self.eval_mode
        bundle_dictionary['training_steps'] = self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
          agent's state.

        Args:
          checkpoint_dir: str, path to the checkpoint saved by tf.Save.
          iteration_number: int, checkpoint version, used when restoring replay
            buffer.
          bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
          bool, True if unbundling was successful.
        """
        try:
            # self._replay.load() will throw a NotFoundError if it does not find all
            # the necessary files, in which case we abort the process & return False.
            self._replay.load(checkpoint_dir, iteration_number)
        except tf.errors.NotFoundError:
            return False
        for key in self.__dict__:
            if key in bundle_dictionary:
                self.__dict__[key] = bundle_dictionary[key]
        # Restore the agent's TensorFlow graph.
        self._saver.restore(self._sess,
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True