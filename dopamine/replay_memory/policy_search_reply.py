from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import math
import os
import pickle

import numpy as np
import tensorflow as tf

import gin.tf
import copy
ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

class OutOfGraphReplayBuffer(object):
    def __init__(self,
               observation_shape,
               stack_size,
               batch_size,
               gamma=0.99,
               extra_storage_types=None,
               observation_dtype=np.float32):
        assert isinstance(observation_shape, tuple)

        tf.logging.info('Creating a %s replay memory with the following parameters:',
            self.__class__.__name__)
        tf.logging.info('\t observation_shape: %s', str(observation_shape))
        tf.logging.info('\t observation_dtype: %s', str(observation_dtype))
        tf.logging.info('\t gamma: %f', gamma)

        self._observation_shape = observation_shape
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._extra_storage_types=extra_storage_types
        self._working_trajectory = None
        self.build_working_trajectory()
        self._trajectory_store = []
        self._stack_size = stack_size
        self._batch_size = batch_size
        self._state_shape = self._observation_shape + (self._stack_size,)

    def get_storage_signature(self):
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
        list of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
            ReplayElement('action', (), np.int32),
            ReplayElement('reward', (), np.float32),
            ReplayElement('terminal', (), np.uint8),
            ReplayElement('baseline',(),np.float32),
            ReplayElement('Gt',(),np.float32)
        ]

        for extra_replay_element in self._extra_storage_types:
            storage_elements.append(extra_replay_element)
        return storage_elements
    def build_working_trajectory(self):
        if self._working_trajectory is not None:
            for name,val in enumerate(self._working_trajectory):
                val.clear()
            return
        self._working_trajectory = {}
        for elements in self.get_storage_signature():
            self._working_trajectory[elements.name] = []

    def add(self, observation, action, reward, terminal, *args):
        self._add(observation, action, reward, terminal, *args)
        if terminal:
            reward_list = self._working_trajectory['reward']
            Gt_list = self._working_trajectory['Gt']
            N = reward_list.size()
            Gt_list[N-1] = reward_list[N-1]
            for i in range(N-1,0,-1):
                Gt_list[i-1] = reward_list[i-1] + Gt_list[i] * self._gamma 
            self._trajectory_store.append(copy.deepcopy(self._working_trajectory))
            self.build_working_trajectory()
    
    def _add(self,*args):
        args_list = []
        for elements in self.get_storage_signature():
            args_list.append(elements.name)
        for name,val in zip(args_list,args):
            self._working_trajectory[name].append(val)

    def sample_transition_batch(self, batch_size=None, indices=None):
        all_data = []
        for elements in self.get_storage_signature():
            ele_data = []
            for traj in self._trajectory_store:
                ele_data += traj[elements.name]
            ele_array = np.array(ele_data[:self._batch_size])
            if elements.name == 'observation':
                ele_array = np.reshape(ele_array,self._observation_shape)
            all_data.append(ele_array)
        self._trajectory_store.clear()
        return tuple(all_data)

    def load(self, checkpoint_dir, suffix):
        pass

    def save(self, checkpoint_dir, iteration_number):
        pass

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement('state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
            ReplayElement('action', (batch_size,), np.int32),
            ReplayElement('reward', (batch_size,), np.float32),
            ReplayElement('next_state', (batch_size,) + self._state_shape,
                          self._observation_dtype),
            ReplayElement('terminal', (batch_size,), np.uint8),
            ReplayElement('indices', (batch_size,), np.int32)
        ]
        for element in self._extra_storage_types:
            transition_elements.append(
                ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                              element.type))
        return transition_elements

@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedReplayBuffer(object):
    def __init__(self,
                 observation_shape,
                 stack_size,
                 use_staging=True,
                 gamma=0.99,
                 wrapped_memory=None,
                 max_sample_attempts=MAX_SAMPLE_ATTEMPTS,
                 extra_storage_types=None,
                 observation_dtype=np.float32):
        if wrapped_memory is not None:
            self.memory = wrapped_memory
        else:
            self.memory = OutOfGraphReplayBuffer(
                observation_shape, gamma,stack_size,
                observation_dtype=observation_dtype,
                extra_storage_types=extra_storage_types)

    def add(self, observation, action, reward, terminal, *args):
        self.memory.add(observation, action, reward, terminal, *args)

    def create_sampling_ops(self, use_staging):
        with tf.name_scope('sample_replay'):
            with tf.device('/cpu:*'):
                transition_type = self.memory.get_transition_elements()
                transition_tensors = tf.py_func(
                    self.memory.sample_transition_batch, [],
                    [return_entry.type for return_entry in transition_type],
                    name='replay_sample_py_func')
                self._set_transition_shape(transition_tensors, transition_type)
                if use_staging:
                    transition_tensors = self._set_up_staging(transition_tensors)
                    self._set_transition_shape(transition_tensors, transition_type)

                # Unpack sample transition into member variables.
                self.unpack_transition(transition_tensors, transition_type)

    def _set_transition_shape(self, transition, transition_type):
        """Set shape for each element in the transition.

        Args:
          transition: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements descriving the shapes of the
            respective tensors.
        """
        for element, element_type in zip(transition, transition_type):
            element.set_shape(element_type.shape)

    def _set_up_staging(self, transition):
        """Sets up staging ops for prefetching the next transition.

        This allows us to hide the py_func latency. To do so we use a staging area
        to pre-fetch the next batch of transitions.

        Args:
          transition: tuple of tf.Tensors with shape
            memory.get_transition_elements().

        Returns:
          prefetched_transition: tuple of tf.Tensors with shape
            memory.get_transition_elements() that have been previously prefetched.
        """
        transition_type = self.memory.get_transition_elements()

        # Create the staging area in CPU.
        prefetch_area = tf.contrib.staging.StagingArea(
            [shape_with_type.type for shape_with_type in transition_type])

        # Store prefetch op for tests, but keep it private -- users should not be
        # calling _prefetch_batch.
        self._prefetch_batch = prefetch_area.put(transition)
        initial_prefetch = tf.cond(
            tf.equal(prefetch_area.size(), 0),
            lambda: prefetch_area.put(transition), tf.no_op)

        # Every time a transition is sampled self.prefetch_batch will be
        # called. If the staging area is empty, two put ops will be called.
        with tf.control_dependencies([self._prefetch_batch, initial_prefetch]):
            prefetched_transition = prefetch_area.get()

        return prefetched_transition

    def unpack_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

        Args:
          transition_tensors: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.transition[element_type.name] = element

        # TODO(bellemare): These are legacy and should probably be removed in
        # future versions.
        self.states = self.transition['state']
        self.actions = self.transition['action']
        self.rewards = self.transition['reward']
        self.next_states = self.transition['next_state']
        self.terminals = self.transition['terminal']
        self.indices = self.transition['indices']

    def save(self, checkpoint_dir, iteration_number):
        """Save the underlying replay buffer's contents in a file.

        Args:
          checkpoint_dir: str, the directory where to read the numpy checkpointed
            files from.
          iteration_number: int, the iteration_number to use as a suffix in naming
            numpy checkpoint files.
        """
        self.memory.save(checkpoint_dir, iteration_number)

    def load(self, checkpoint_dir, suffix):
        """Loads the replay buffer's state from a saved file.

        Args:
          checkpoint_dir: str, the directory where to read the numpy checkpointed
            files from.
          suffix: str, the suffix to use in numpy checkpoint files.
        """
        self.memory.load(checkpoint_dir, suffix)