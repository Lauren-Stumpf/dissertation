



import threading
import numpy as np



from general_util import LinearSchedule
import tensorflow as tf
import collections
import random

import math

    

def decompress_transition_list(transitions_list):
        states = tf.cat([t.s for t in transitions_list])
        actions = tf.cat([t.a for t in transitions_list])
        rewards = tf.cat([t.r for t in transitions_list])
        next_states = tf.cat([t.next_s for t in transitions_list if t.next_s is not None])
        mask = tf.tensor([not t.terminal for t in transitions_list])
        td_errors = tf.cat([t.td_error for t in transitions_list if t.td_error is not None])
        action_probs = tf.cat([t.action_prob for t in transitions_list if t.action_prob is not None])
        return states, actions, rewards, next_states, mask, td_errors, action_probs




class Trajectory(list):

    def terminal(self):
        return self[-1].terminal
    
    def decompress(self):
        return decompress_transition_list(self)  

    def compress(self, s, e):
        return self[s:e]



class StructuredReplayBuffer(collections.deque):

    def __init__(self, capacity):
        super().__init__([], maxlen=int(capacity))
    
    def sample(self, number_of_samples):
        indices = random.sample(range(len(self)), number_of_samples)
        return [self[idx] for idx in indices]
    
    def to_tensor(elements):
        if len(elements) == 0:
            return None
        return tf.cat(elements)
    
    def decompress(self):
        return decompress_transition_list(self)
    
    

class SumTree(object):
    def __init__(self, capacity):
        self.d, self.position, self.capacity = 0, 0, capacity
        self.data = [None for i in range(self.capacity)]
        self.depth = math.ceil(math.log(capacity+1, 2))+1
        self.length = 2**self.depth-1
        self.tree = [0 for i in range(self.length)]

    
    def get_val(self, index):
        number = 2**(self.depth-1)-1+index
        return self.tree[number]

            
    def update_tree(self, index, value):
        number = 2 ** (self.depth-1) - 1 + index
        distinction = value - self.tree[number]
        self.rebuild(number, distinction)
    
    def find(self, value, bool_normalise=True):
        if bool_normalise:
            value *= self.tree[0]
        return self.subclass_find(value, 0)

    def subclass_find(self, value, index):
        if 2**(self.depth-1)-1 <= index:
            return self.data[index-(2**(self.depth-1)-1)], self.tree[index], index-(2**(self.depth-1)-1)


        if value <= self.tree[2*index+1]:
            return self.subclass_find(value,2*index+1)
        
    
        return self.subclass_find(value-self.tree[2*index+1],2*(index+1))
        
    def rebuild(self, idx, distinction):
        self.tree[idx] += distinction
        if not idx == 0:
            idx = int((idx-1)/2)
            self.rebuild(idx, distinction)

    def capacity(self):
        return self.d
    
    def addition(self, contents, value):
        index = self.position
        self.position, self.d = (self.position+1) % self.capacity, min(self.d+1, self.capacity)
        self.data[index] = contents
        self.update_tree(index, value)
        

#Adapted from Mutual Information State Intrinsic Control 
class Buffer_Structure:
    def __init__(self, capacities, dims, time, samples_processed_for_replay_buffer, mutual_information_use_in_buffer):
  
        self.cur_capacity, self.number_interactions = 0, 0

        
        
        self.d = dims // time

        self.storage_structure = {val: np.empty([self.d, *dim]) for val, dim in capacities.items()}
       
        
        self.storage_structure['s'], self.storage_structure['m']  = np.empty([self.d, 1]), np.empty([self.d, 1])
        
    

        self.storage_dim, self.sampling_function, self.time =  capacities, samples_processed_for_replay_buffer, time


        self.freeze = threading.Lock()

    

    def return_current_size(self):
        with self.freeze:
            return self.cur_capacity * self.time

    def return_trajectories(self, ddpg, ir, batch_size, mutual_information_normalization_coefficient, learn_multiple_skills, time):

        structures = {}

        with self.freeze:
            for val in self.storage_structure.keys():
                structures[val] = self.storage_structure[val][:self.cur_capacity]

        structures['ag_2'] = structures['ag'][:, 1:, :]
        structures['o_2'] = structures['o'][:, 1:, :]


        transitions = self.sampling_function(ddpg, ir, structures, batch_size, mutual_information_normalization_coefficient, learn_multiple_skills, time)



        return transitions
    
    def reset_replay_buffer(self):
        with self.freeze:
            self.cur_capacity = 0

    def add_interaction_to_replay_buffer(self, episode_batch, ddpg):
       
        len_trajectory = [len(episode_batch[key]) for key in episode_batch.keys()]
        


        with self.freeze:
            index = self.return_indices_of_samples(len_trajectory[0]) 
          
            for val in self.storage_structure.keys():
                self.storage_structure[val][index] = episode_batch[val]

            self.number_interactions =  self.number_interactions + len_trajectory[0] * self.time

    

    def get_transitions_stored(self):
        with self.freeze:
            return self.number_interactions

    

    def return_indices_of_samples(self, increment=None):
        increment = increment or 1  
        
        if self.cur_capacity+increment <= self.d:
            index = np.arange(self.cur_capacity, self.cur_capacity+increment)
            
        elif self.cur_capacity < self.d:
            excess = increment - (self.d - self.cur_capacity)
            limit_index = np.arange(self.cur_capacity, self.d)
            cur_index = np.random.randint(0, self.cur_capacity, excess)
            index = np.concatenate([limit_index, cur_index])
            
        else:
            index = np.random.randint(0, self.d, increment)

 
        self.cur_capacity = min(self.d, self.cur_capacity+increment)

        if increment == 1:
            index = index[0]
        return index
    
    


class HER_Sample_Strategy:
    def __init__(self, her_buffer_strategy, her_replay_parameter, reward_func=None):
        self.her_buffer_strategy = her_buffer_strategy
        self.her_replay_parameter = her_replay_parameter
    

    def sample_her_transitions(self, sample, trajectory_length):
        timestep, sample_size  = sample['actions'].shape[1], sample['actions'].shape[0]

        sampled_index, timestep_increment = np.random.randint(0, sample_size, trajectory_length), np.random.randint(timestep, size=trajectory_length)
        trajectory = {ep: sample[ep][sampled_index, timestep_increment].copy() for ep in sample.keys()}

        her_index = np.where(np.random.uniform(size=trajectory_length) < self.future_p)
        offset = np.random.uniform(size=trajectory_length) * (timestep - timestep_increment)
        
        
        offset = offset.astype(int)
        next_timestep = (timestep_increment + 1 + offset)[her_index]

        achieved_goal = sample['ag'][sampled_index[her_index], next_timestep]
        trajectory['g'][her_index] = achieved_goal

        trajectory['r'] = np.expand_dims(self.reward_func(trajectory['ag_next'], trajectory['g'], None), 1)
        trajectory = {k: trajectory[k].reshape(trajectory_length, *trajectory[k].shape[1:]) for k in trajectory.keys()}

        return trajectory
    
    
#Adapted from Mutual Information State Intrinsic Control
def her_function(her_buffer_strategy, her_replay_parameter, return_reward, schedule_for_mutual_information, weight_scheduler, mutual_information_use_in_buffer):
    
   
    MI_schedule = LinearSchedule(endpoints=schedule_for_mutual_information)
    skill_schedule = LinearSchedule(endpoints=weight_scheduler)

    def function_that_reframes_goal(agent, intrinsic_reward, episode_samples_tf, interactions_size, mutual_information_normalization_coefficient, learn_multiple_skills, t):
        
      
        index, index_time = return_random_sampling_strategy(episode_samples_tf['u'].shape[0], interactions_size, episode_samples_tf['u'].shape[1])
       
        MI_transition, obs_transition = np.zeros([index.shape[0], 1]), np.zeros([index.shape[0], 1])
  
        if intrinsic_reward:
            
            current_obs = episode_samples_tf['o'][index, index_time].copy()
            suceeding_obs = episode_samples_tf['o'][index, index_time+1].copy()

            current_obs = np.reshape(current_obs, (current_obs.shape[0], 1, current_obs.shape[-1]))
            suceeding_obs = np.reshape(suceeding_obs, (suceeding_obs.shape[0], 1, suceeding_obs.shape[-1]))
            
            current_and_future_obs = np.concatenate((current_obs, suceeding_obs), axis=1)

            if mutual_information_normalization_coefficient > 0:
                loss = agent.run_mi(current_and_future_obs)
                MI_transition = (-loss).copy()


        h_ind = np.where(np.random.uniform(size=interactions_size) < 1 - (1. / (1 + her_replay_parameter)))
        length_of_traj = np.random.uniform(size=interactions_size) * (episode_samples_tf['u'].shape[1] - index_time)
        length_of_traj = length_of_traj.astype(int)
        t_ind = (index_time + 1 + length_of_traj)[h_ind]        

        interactions = {}
        for val in episode_samples_tf.keys():
            if not (val == 'm' or val == 's' or val == 'p'):
                interactions[val] = episode_samples_tf[val][index, index_time].copy()
            else:
                interactions[val] = episode_samples_tf[val][index].copy()
                
        
        interactions['s'] = interactions['s'].flatten().copy()
        interactions['m'] = interactions['m'].flatten().copy()

        


        
        interactions['g'][h_ind] = episode_samples_tf['ag'][index[h_ind], t_ind]

        info = {}
        for val, item in interactions.items():
            if val.startswith('info_'):
                info[val.replace('info_', '')] = item

        r_data = {i: interactions[i] for i in ['ag_2', 'g']}
        r_data['info'] = info
        interactions['r'] = return_reward(**r_data)

        interactions = {i: interactions[i].reshape(interactions_size, *interactions[i].shape[1:]) for i in interactions.keys()}

        if intrinsic_reward:
            interactions['m'] = MI_transition.flatten().copy()
            interactions['s'] = obs_transition.flatten().copy()

        
        interactions['e_w'], interactions['s_w'], interactions['r_w'], interactions['m_w']  = skill_schedule.value(t), 1.0, 1.0, MI_schedule.value(t)
      

        return interactions

    return function_that_reframes_goal


def return_random_sampling_strategy(lower_range, upper_range, number):
    index = np.random.randint(0, lower_range, upper_range)
    index_time = np.random.randint(number, size=upper_range)
    return index, index_time




