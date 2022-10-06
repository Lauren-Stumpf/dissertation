



import numpy as np

import tensorflow as tf
from mpi4py import MPI

import threading

#Built upon from Mutual Information State Intrinsic Control 
class Clip_And_Standardise:
    def __init__(self, number_of_samples_to_process, delta, bounds, session):

        
        
        dtype, to_train, init, np_dtype = tf.float32, False, tf.zeros_initializer(), np.float32
        
        self.batch_dim = number_of_samples_to_process
        self.delta = delta
        self.upper_and_lower_bounds = bounds
        self.session = session if session is not None else tf.get_default_session()

        self.summation_obs, self.squared_obs, self.freq_obs  = np.zeros(self.batch_dim, np_dtype), np.zeros(self.batch_dim, np_dtype), np.zeros(1, np_dtype)
        
        self.action_space_high, self.action_space_low = 1, 1
        
        self.addition_var = tf.get_variable(initializer = init, shape=self.summation_obs.shape, name='addition',trainable = to_train, dtype = dtype)
        self.square_var = tf.get_variable(initializer = init, shape=self.squared_obs.shape, name='squared',trainable = to_train, dtype = dtype)
        self.avg_var = tf.get_variable(initializer = init, shape=(self.batch_dim,), name='avg_var',trainable = to_train, dtype = dtype)
        
        init = tf.ones_initializer()
        
        self.freq = tf.get_variable(initializer = init, shape=self.freq_obs.shape, name='freq',trainable = to_train, dtype = dtype)
        self.sigma = tf.get_variable(initializer = init, shape=(self.batch_dim,), name='sigma',trainable = to_train, dtype = dtype)
        self.freq_squashed = tf.placeholder(name='freq_squashed', shape=(1,), dtype = dtype)
        self.addition_squashed = tf.placeholder(name='addition_squashed', shape=(self.batch_dim,), dtype = dtype)
        self.squared_squashed = tf.placeholder(name='squared_squashed', shape=(self.batch_dim,), dtype = dtype)

        self.amend_op = tf.group(self.freq.assign_add(self.freq_squashed),self.addition_var.assign_add(self.addition_squashed),self.square_var.assign_add(self.squared_squashed))
        self.recalculate_op = tf.group(tf.assign(self.avg_var, self.addition_var / self.freq),tf.assign(self.sigma, tf.sqrt(tf.maximum(tf.square(self.delta),self.square_var / self.freq - tf.square(self.addition_var / self.freq)))),)
        self.freeze = threading.Lock()
        
    
    def standardize(self, x):
        bound = self.upper_and_lower_bounds
 
        shape = ([1] * (len(x.get_shape())-1)) + [-1]
        avg = tf.reshape(tf.cast(self.avg_var, x.dtype), shape)
        standard_dev = tf.reshape(tf.cast(self.sigma, x.dtype), shape)
        normalised = (x - avg) / standard_dev
        clipped_normalised = tf.clip_by_value(normalised, -bound, bound)
        
        return clipped_normalised
    
    def transform_stats(self):
        
        add = self.map_to(self.addition_var)
        squared = self.map_to(self.square_var)
        avg = self.map_to(self.avg_var)
        freq = self.map_to(self.freq)
        standard_dev = self.map_to(self.sigma)
        freq_squashed = self.map_to(self.freq_squashed)
        add_squashed = self.map_to(self.addition_squashed)
        squared_squashed = self.map_to(self.squared_squashed)
        
        return add, squared, avg, freq, standard_dev, freq_squashed, add_squashed, squared_squashed
    
    def amend(self, x):
        x = x.reshape(-1, self.batch_dim)

        with self.freeze:
            self.summation_obs = self.summation_obs + x.sum(axis=0)
            self.squared_obs = self.squared_obs + (np.square(x)).sum(axis=0)
            self.freq_obs[0] = self.freq_obs[0] + x.shape[0]
    
    def reshape_for_broadcasting(source, target):

        dim = len(target.get_shape())
        shape = ([1] * (dim-1)) + [-1]
        return tf.reshape(tf.cast(source, target.dtype), shape)
    
    def denormalise_action(self, action):
        action -= self.action_space_low
        
        action /= (self.action_space_high - self.action_space_low)
        
        action = action * 2 - 1
        return action

    def destandardize(self, x):
        
        shape = ([1] * (len(x.get_shape())-1)) + [-1]
        avg = tf.reshape(tf.cast(self.avg_var, x.dtype), shape)
        standard_dev = tf.reshape(tf.cast(self.sigma, x.dtype), shape)
        denormalised = avg + x * standard_dev
        
        return denormalised

    def synchronize(self, total, squared_total, freq, root=None):
        total[...] = self._mpi_average(total)
        squared_total[...] = self._mpi_average(squared_total)
        freq[...] = self._mpi_average(freq)
        return total, squared_total, freq
    
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf
    
    def get_reward(self, reward_list, normalized_reward=True, scaling = True, theta = 0.2):
        processed_rewards = []
        if normalized_reward:
    
            for step in range(len(reward_list)):
                
                if scaling == True:
                    r_i, r_e = reward_list[step]
                    rew = theta * r_i + (1 - theta) * r_e
                    n_reward = self.normalise(rew)
                    
                else:
                    n_reward = self.normalise(reward_list[step])
                
                processed_rewards.append(n_reward)
                
        else:
            return reward_list
		
        return np.array(processed_rewards)

    def recalculate(self):
        with self.freeze:
            freq = self.freq_obs.copy()
            self.freq_obs[...] = 0
            
            total = self.summation_obs.copy()
            self.summation_obs[...] = 0
            
            squared_total = self.squared_obs.copy()
            self.squared_obs[...] = 0

        total_s, squared_s, freq_s = self.synchronize(total=total, squared_total=squared_total, freq=freq)

        self.session.run(self.amend_op, feed_dict={self.freq_squashed: freq_s,self.addition_squashed: total_s,self.squared_squashed: squared_s,}) # could remove comma
        self.session.run(self.recalculate_op)
        
    def normalise_action(self, action):
        action = (action + 1) / 2 
        
        action *= (self.action_space_high - self.action_space_low)
        
        action += self.action_space_low
        return action

    
    
    def map_to(self, observation):
        with self.freeze:
            max_ob = max(observation)
            min_ob = min(observation)
            
            n_ob = (observation - min_ob) / (max_ob - min_ob)
        
        return n_ob
    

    
    
   

        
