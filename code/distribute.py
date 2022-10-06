

from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from general_util import convert_episode_to_batch_major, store_args

#Build upon from Mutual Information State Intrinsic Control 

class Distribute_Workers_Class:

    @store_args
    def __init__(self, make_env, agent, space_depth, communicator, T, rollout_batch_size=1,
                 optimise=False, double_Q_learning=False, return_Q_values_reg=False, noise_constant=0,
                 small_constant_for_exploration=0, history_len=100, render=False, **kwargs):
        
        first_dim, second_dim, dtype, max_len = self.rollout_batch_size, self.space_depth, np.float32, history_len

        self.number_of_trajectories = 0
        self.list_of_environments= [make_env() for i in range(rollout_batch_size)]
        
        self.goal_rollout = np.empty((first_dim, second_dim['g']), dtype)  
        self.first_observation = np.empty((first_dim, second_dim['o']), dtype)  
        self.first_achieved_goal = np.empty((first_dim, second_dim['g']), dtype)  
        
        
        self.trajectories_with_goal_made = deque(maxlen=max_len)
        self.stack_of_Q_values = deque(maxlen=max_len)

        self.info_keys = [val.replace('info_', '') for val in space_depth.keys() if val.startswith('info_')]

        
        self.reset_all_workers()
        self.reset_trajectories()
        
    def seed(self, s):
        
        for i, e in enumerate(self.list_of_environments):
            e.seed(s + 1000 * i)
        
    def mean_Q_across_workers(self):
        return wrapper_mean(self.stack_of_Q_values)
        
    def success_across_all_workers(self):
        return wrapper_mean(self.trajectories_with_goal_made)


    def set_up_workers(self, generated_goal=False, z_s_onehot=False):

        self.reset_all_workers(generated_goal)

        first_dim, second_dim, dtype = self.rollout_batch_size, self.space_depth, np.float32
        
        
        list_of_observations = []
        list_of_skills = []
        list_of_achieved_goals = []
        list_of_actions = []
        list_of_goals = []
        list_of_successful_traj = []

        
        if self.render == 'rgb_array':
            imgs = np.empty([first_dim, self.T, 500, 500, 3])
        elif self.render == 'human':
            imgs = np.empty([first_dim, self.T, 992, 1648, 3])
            
        info_values = [np.empty((self.T, first_dim, second_dim['info_' + val]), dtype) for val in self.info_keys]
        Q_values = []
        
        observation_rollout, achieved_goals_rollout = np.empty((first_dim, second_dim['o']), dtype) , np.empty((first_dim, second_dim['g']), dtype) 
        observation_rollout[:], achieved_goals_rollout[:] = self.first_observation, self.first_achieved_goal
   
        
        skill_rollout = z_s_onehot.copy()  
        
        
        for t in range(self.T):
            policy_output = self.agent.get_actions(observation_rollout, skill_rollout, achieved_goals_rollout, self.goal_rollout, compute_Q=self.return_Q_values_reg,
                noise_constant=self.noise_constant if not self.optimise else 0.,
                small_constant_for_exploration=self.small_constant_for_exploration if not self.optimise else 0.,
                use_target_net=self.double_Q_learning)

            if not self.return_Q_values_reg:
                u = policy_output
                
            else:
                u, Q = policy_output
                Q_values.append(Q)

            if u.ndim == 1:
                u = u.reshape(1, -1)

            new_observation, new_achieved_goal, success = np.empty((first_dim, second_dim['o'])), np.empty((first_dim, second_dim['g'])), np.zeros(first_dim)
            
            
            for i in range(first_dim):
                try:
                    
                    current_new_observation, _, _, info = self.list_of_environments[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                      
                    for index, val in enumerate(self.info_keys):
                        info_values[index][t, i] = info[val]
                    
                    new_achieved_goal[i] = current_new_observation['achieved_goal']
                    new_observation[i] = current_new_observation['observation']
                    
                    if self.render:
                        if self.render == 'rgb_array':
                            imgs[i][t] = self.list_of_environments[i].render(mode='rgb_array')
                        elif self.render == 'human':
                            imgs[i][t] = self.list_of_environments[i].render()

                except MujocoException as e:
                    return self.set_up_workers()

            

            list_of_observations.append(observation_rollout.copy())
            list_of_actions.append(u.copy())
            
            list_of_successful_traj.append(success.copy())
            list_of_achieved_goals.append(achieved_goals_rollout.copy())
            
            list_of_skills.append(skill_rollout.copy())
            list_of_goals.append(self.goal_rollout.copy())
            observation_rollout[...] = new_observation
            achieved_goals_rollout[...] = new_achieved_goal
            
        
        list_of_achieved_goals.append(achieved_goals_rollout.copy())
        list_of_observations.append(observation_rollout.copy())
        self.first_observation[:] = observation_rollout


        episode = dict(o=list_of_observations,z=list_of_skills,u=list_of_actions,g=list_of_goals,ag=list_of_achieved_goals,)
        for i, j in zip(self.info_keys, info_values):
            episode['info_{}'.format(i)] = j

        successful = np.array(list_of_successful_traj)[-1, :].copy()
        
        success_rate = np.mean(successful)
        self.trajectories_with_goal_made.append(success_rate)
        
        
        if self.return_Q_values_reg:
            self.stack_of_Q_values.append(np.mean(Q_values))
            
            
        self.number_of_trajectories =  self.number_of_trajectories + self.rollout_batch_size

        if self.render == 'rgb_array' or self.render == 'human':
            return imgs, convert_episode_to_batch_major(episode)
        else:
            return convert_episode_to_batch_major(episode)

    def reset_trajectories(self):
        self.trajectories_with_goal_made.clear()
        self.stack_of_Q_values.clear()
        
    def gather(self, data, index_2d):

        batch_size = data.shape[0]
        data_dim = data.shape[2]
        index_len = index_2d.shape[1]
 
        res = data[np.arange(batch_size).repeat(np.array([index_len])),index_2d.flatten()].reshape(batch_size, index_len, data_dim)
        return res

    def reset_all_workers(self, generated_goal=False):

        for i in range(self.rollout_batch_size):
            self.reset_rollout(i, generated_goal)
            
    def split_sequence_tensor(self, x, num_steps):
    
        keys = x.keys()
        has_weights = x._weights is not None
        split_dict = {}
        for i in range(num_steps):
            split_dict[i] = {}
        for key in keys:
            keyed_x = x[key]
            weights, values, lengths = keyed_x._weights, keyed_x.values(), keyed_x.lengths()

            splitted_values = np.split(values, lengths.tolist())
            if has_weights:
                splitted_weights =np.split(weights, lengths.tolist())
                
            for i in range(num_steps):
                split_dict[i][key] = (lengths[i::num_steps],np.concatenate(splitted_values[i::num_steps]),
                    np.concatenate(splitted_weights[i::num_steps]) if has_weights else None,
                )
    
        result = []
        for i in range(num_steps):
            result.append(split_dict[i])
            
        return result
    
    def check_nan(self, data):

        for i in range(len(data)):
            value, presence = data[i]
            value, presence = value, presence
            presence[np.isnan(value)] = 0
            value[np.isnan(value)] = 0
            data[i] = (value, presence).float()

        return data


    def _build_id_2_embedding_size(keys, feature_configs, id_mapping_configs):

        id_2_embedding_size = {}
        for key, feature_config, id_mapping_config in zip(
            keys, feature_configs, id_mapping_configs
        ):
            id_2_embedding_size[key] = {
                config.feature_id: id_mapping_config[
                    config.id_mapping_name
                ].embedding_table_size
                for config in feature_config
            }
        return id_2_embedding_size
    

    def save_policy(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.agent, f)
            
    def stack_(self, mems):
        with self.lock:
            shape = (-1, *mems[0].shape)
            return np.concatenate(mems).reshape(*shape)

    def logs(self, prefix='worker'):
        
        logs = []
        logs += [('success_rate', np.mean(self.trajectories_with_goal_made))]
        if self.return_Q_values_reg:
            logs += [('mean_Q', np.mean(self.stack_of_Q_values))]
        logs += [('episode', self.number_of_trajectories)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
        
    def _dense_to_sparse(self, dense):
    
        return [{i: v.item() for i, v in enumerate(elem)} for elem in dense]
        
    def reset_rollout(self, i, generated_goal):
  
        list_of_observations = self.list_of_environments[i].reset()
        self.goal_rollout[i] = list_of_observations['desired_goal']
        if isinstance(generated_goal, np.ndarray):
            self.goal_rollout[i] = self.list_of_environments[i].env.goal = generated_goal[i].copy()
        self.first_observation[i], self.first_achieved_goal[i]  = list_of_observations['observation'], list_of_observations['achieved_goal']


def wrapper_mean(np_array):
    try:
        return np.mean(np_array)
    except:
        return None

