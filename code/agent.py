


import tensorflow as tf
import numpy as np
from collections import OrderedDict
from tensorflow.contrib.staging import StagingArea
import logger
from general_util import (
    import_function, store_args, flatten_grads,  load_weight)
from format_file import Clip_And_Standardise
from experience_replay import Buffer_Structure
from MPI_util import MpiAdam
import json
from collections import deque
import random 

import math
import statistics






#Adapted from StableBaselines
def new_nn(input_, list_of_layers,  reuse=None, flatten=False, name="", set_progression = 1, name_is = '',    type_ = '', dropout = True,  Recurrent = True, D2RL = True, initial_state = True):
    first_input = input_
    next_input = input_
    number_of_layers = len(list_of_layers)
    name = name_is + name
    if set_progression != 1:
        if Recurrent == True:
    
          next_input = tf.expand_dims(next_input, axis=2)
          lstm = tf.compat.v1.keras.layers.LSTM(units = list_of_layers[0], name = name + str(1000), return_state=True, stateful = True)
          next_input = lstm(next_input)
    
        for i, size in enumerate(list_of_layers):
            act = tf.nn.relu if i < len(list_of_layers)-1 else None
    
            if i == number_of_layers - 1 and type_ == 'policy': 
              kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.001) 
    
            else: 
              kernel_initializer  = tf.contrib.layers.xavier_initializer()
    
    
            x = tf.layers.dense(inputs=next_input, units=size, kernel_initializer=kernel_initializer, reuse=reuse, name=name+'_'+str(i))
    
            if D2RL == True: 
              next_input = tf.concat(values = [x, first_input], axis=1)
    
            else: 
              next_input = x
            
            if act:
                x = act(x)
    
        if dropout == True: 
          x = tf.nn.dropout(x, rate = 0.5, name = name + str(2021))
    
        if flatten:
    
            x = tf.reshape(x, [-1])
            
        return x
    
    for i, size in enumerate(list_of_layers):
        act = tf.nn.relu if i < len(list_of_layers)-1 else None
        input_ = tf.layers.dense(inputs=input_,units=size,kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse,name=name+'_'+str(i))
        if act:
            input_ = act(input_)
    if flatten:

        input_ = tf.reshape(input_, [-1])
    return input_
    






def td_target(reward, discount, next_value):
    return reward + discount * next_value

#Adapted from StableBaselines
def huber_loss(x):
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, (tf.abs(x) - 0.5))

#Adapted from StableBaselines
def likelihood(x, mean, standard_dev_logarithm):
    pre_sum = -0.5 * (((x-mean)/(tf.exp(standard_dev_logarithm)+ 1e-8 ))**2 + 2*standard_dev_logarithm + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

#Adapted from StableBaselines
def clip_gradient(x, lower_bound=-1., upper_bound=1.):
    clip_up = tf.cast(x > upper_bound, tf.float32)
    clip_low = tf.cast(x < lower_bound, tf.float32)
    return x + tf.stop_gradient((upper_bound - x)*clip_up + (lower_bound - x)*clip_low)


def additive_noise(x, annealing_probability_threshold, alpha, beta):
    
    add_noise = np.random.normal(0,1,1)
    
    multiply_noise = random.uniform(alpha, beta)
    
    if random.uniform(0, 1) > annealing_probability_threshold:
        n = random.randint(0, 1)
        if n == 0:
            x += add_noise
        else:
            x = x * multiply_noise
            
    return x 
    
    
    
def detractive_noise(x, annealing_probability_threshold, rho, lambda_):
    
    subtractive_noise = np.random.normal(0,1,1)
    
    division_noise = random.uniform(rho, lambda_)
    
    if random.uniform(0, 1) > annealing_probability_threshold:
        n = random.randint(0, 1)
        if n == 0:
            x -= subtractive_noise
        else:
            x = x / division_noise
            
    return x 

#Adapted from StableBaselines
def policy(x, dimension_of_action_space, intermediate_layers, list_of_layers, name_is = ''):
    if '_' in name_is:
        reuse = tf.AUTO_REUSE
        
    else:
        reuse = None
        
    if reuse != None :
        net = new_nn(x, [intermediate_layers] * (list_of_layers+1), reuse = reuse)
    else:
        net = new_nn(x, [intermediate_layers] * (list_of_layers+1))
    
    if reuse != None:
        mean = tf.layers.dense(net, dimension_of_action_space, activation=None, reuse = reuse)
    
    else:
        mean = tf.layers.dense(net, dimension_of_action_space, activation=None)

    standard_dev_logarithm = tf.layers.dense(net, dimension_of_action_space, activation=tf.tanh)
    
    standard_dev_logarithm = -5 + 0.5 * (2 - -5) * (standard_dev_logarithm + 1)

    std = tf.exp(standard_dev_logarithm)
    
    pi = mean + tf.random_normal(tf.shape(mean)) * std
    
    logp_pi = likelihood(pi, mean, standard_dev_logarithm)
    return mean, pi, logp_pi

def reparameterise(mean, prob, probability_logarithm):
    mean = tf.tanh(mean)
    
    prob = tf.tanh(prob)
    
    probability_logarithm -= tf.reduce_sum(tf.log(clip_gradient(1 - prob**2, lower_bound =0, upper_bound=1) + 1e-6), axis=1)
    return mean, prob, probability_logarithm

#Adapted from StableBaselines
class Policy_Value_Function_Structure:
    @store_args
    def __init__(self, interaction_inp, observation_space_dimension, goal_space_dimension, action_space_dimension, maximum_action_value, o_stats, g_stats, network_width, layers, stochastic, **kwargs):


        self.goal_policy_value, self.skill_policy_value, self.action_policy_value, self.observation_policy_value = interaction_inp['g'], interaction_inp['z'], interaction_inp['u'], interaction_inp['o']

        observation = self.o_stats.standardize(self.observation_policy_value)
        goal = self.g_stats.standardize(self.goal_policy_value)
        skill = self.skill_policy_value

        input_pi = tf.concat(axis=1, values=[observation, skill, goal]) 

        name_is = ''
        stochastic, mode = 'TQC', 1
        
        if mode == 1: 
            with tf.variable_scope('pi'):
                mean, prob, probability_logarithm = policy(input_pi, self.action_space_dimension, self.network_width, self.layers)
                mean, prob, self.probability_logarithm_reparameterised = reparameterise(mean, prob, probability_logarithm)
             
                self.mean_reparameterised = mean * self.maximum_action_value
                self.policy_probability = prob * self.maximum_action_value
                self.negative_log_probability_policy = - self.probability_logarithm_reparameterised
        
        elif mode == 2:
            with tf.variable_scope('pi'):
                mean, pretanh, probability_logarithm = policy(input_pi, self.action_space_dimension, self.network_width, self.layers)
                mean, prob, self.probability_logarithm_reparameterised = reparameterise(mean, pretanh, probability_logarithm)
                self.mean_reparameterised = mean * self.maximum_action_value
                self.policy_probability = prob * self.maximum_action_value
                self.negative_log_probability_policy = - self.probability_logarithm_reparameterised
                
        elif mode == 3: 
            with tf.variable_scope('pi'):
                self.policy_probability = self.maximum_action_value * tf.tanh(new_nn(input_pi, [self.network_width] * self.layers + [self.action_space_dimension]))
        
        elif mode == 4: 
            with tf.variable_scope('pi'):
                self.policy_probability = self.maximum_action_value * tf.tanh(new_nn(input_pi, [self.network_width] * self.layers + [self.action_space_dimension]))
        
        else:
            raise Exception('Mode not recognised')


        if mode == 1:
            with tf.variable_scope('Q'):
               
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.policy_probability / self.maximum_action_value])
                self.Q_probability = new_nn(input_to_critic, [self.network_width] * self.layers + [1])
                
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.action_policy_value / self.maximum_action_value])
                self.Q_values_policy_value = new_nn(input_to_critic, [self.network_width] * self.layers + [1], reuse=True)
                
        elif mode == 2:
            with tf.variable_scope('Q'): 
                
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.policy_probability / self.maximum_action_value])
                nets = [new_nn(input_to_critic, [self.network_width] * self.layers + [1], name = str(i), name_is = name_is) for i in range(self.num_quantiles)]
                quantiles = tf.stack(tuple(nets), axis=1)
                self.Q_probability = quantiles
                
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.action_policy_value / self.maximum_action_value])
                nets = [new_nn(input_to_critic, [self.network_width] * self.layers + [1], reuse=True, name = str(i), name_is = name_is) for i in range(self.num_quantiles)]
                quantiles = tf.stack(tuple(nets), axis=1)
                self.Q_values_policy_value = quantiles
        
        elif mode == 3:
            with tf.variable_scope('Q'):  
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.policy_probability / self.maximum_action_value])
                self.Q_probability = new_nn(input_to_critic, [self.network_width] * self.layers + [1], name= str(1), name_is = name_is)
                self.Q_probability_1 = new_nn(input_to_critic, [self.network_width] * self.layers + [1], name= str(2), name_is = name_is)
             
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.action_policy_value / self.maximum_action_value])
            
                self.Q_values_policy_value = new_nn(input_to_critic, [self.network_width] * self.layers + [1], reuse=True, name=str(2), name_is = name_is)
                self.Q_values_policy_value_1 = new_nn(input_to_critic, [self.network_width] * self.layers + [1], reuse=True, name=str(2), name_is = name_is)
        
        elif mode == 4: 
            with tf.variable_scope('Q'):
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.policy_probability / self.maximum_action_value])
                self.Q_probability = new_nn(input_to_critic, [self.network_width] * self.layers + [1])
        
                input_to_critic = tf.concat(axis=1, values=[observation, skill, goal, self.action_policy_value / self.maximum_action_value]) 
                self.Q_values_policy_value = new_nn(input_to_critic, [self.network_width] * self.layers + [1], reuse=True)
    
    def return_global_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


    def return_trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


    def return_perturbable_vars(self):
        return [v for v in self.trainable_vars if 'LayerNorm' not in v.name]
    
    def sac_q_update(self, current_states, actions, rewards, next_states, ends):

        with tf.GradientTape() as tape:
            q = self.Q_values_policy_value(current_states, actions)

            pi, log_pi = self.policy_probability(next_states)
            
            target_1 = self.target_q2(next_states, pi)
            target_2 = self.target_q1(next_states, pi)
            
            min_target = tf.minimum(target_1, target_2)
            factor = self.alpha * log_pi
            target = min_target - factor
            
            rew = self.gamma * ends * target
            q_ = tf.stop_gradient(rewards + rew)
            q_q = (q - q_) ** 2
            critic_1_loss = tf.reduce_mean( q_q )

        with tf.GradientTape() as tape_:
            
            q = self.Q_values_policy_value(current_states, actions)
            
            pi, log_pi = self.policy_probability(next_states)
            
            target_1 = self.target_q1(next_states, pi)
            target_2 = self.target_q2(next_states, pi)
            
            min_target = tf.minimum(target_2, target_1)
            factor = self.alpha * log_pi
            target = min_target - factor
            
            rew = self.gamma * ends * target
            q_ = tf.stop_gradient(rewards + rew)
            q_q = (q - q_)**2
            
            
            critic_2_loss = tf.reduce_mean(q_q)

        grads_1 = tape.gradient(critic_1_loss, self.q1.trainable_variables)
        grads_2 = tape_.gradient(critic_2_loss, self.q2.trainable_variables)
        
        
        
        self.critic_1_optimizer.apply_gradients(zip(grads_1, self.q1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(grads_2, self.q2.trainable_variables))

        

        return critic_1_loss, critic_2_loss

    def update_policy_network_sac(self, current_states):
        with tf.GradientTape() as tape:
            
            pi, log_pi = self.policy_probability(current_states)
            q1 = self.Q_values_policy_value(current_states, pi)
            q2 = self.Q_values_policy_value(current_states, pi)
            q = tf.minimum(q1, q2)

            factor = self.alpha * log_pi
            q = q - factor

            actor_loss = -tf.reduce_mean(q)

        
        grads = tape.gradient(actor_loss, self.policy_probability.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.policy_probability.trainable_variables))

        
        return actor_loss


    def update_alpha(self, current_states):
       with tf.variable_scope('Q') as grad:
            pi, log_pi = self.policy(current_states)
            
            loss = - self.alpha * (log_pi + self.target_entropy)
            alpha_loss = tf.reduce_mean( loss)

       grad = grad.gradient(alpha_loss, [self.alpha])
       self.alpha_optimizer.apply_gradients(zip(grad, [self.alpha]))
       
       return alpha_loss
        
    def return_action(self, state):
        
        mu, std = self.probability_logarithm_reparameterised(state)
        
        
        sigma = tf.exp(std)

     
        dist = likelihood(state, mu, sigma, mode = 1) 
        action_ = dist.sample()

      
        action = tf.tanh(action_)

  
        log_pi_ = dist.log_prob(action_)
        log_pi = log_pi_ - tf.reduce_sum(tf.math.log(1 - action**2 +  1e-16), axis=1,keepdims=True)

        return action, log_pi
    
    def TD3_critic_call(self, obs,action, reuse=False):
        
        with tf.variable_scope('Q', reuse=tf.AUTO_REUSE):
            v = self.network_builder(obs)
            v = tf.layers.dense(v, action)
            v = tf.nn.tanh(v)
            
        return v
    
    def TD3_actor_call(self, obs, action, reuse=False):
        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):

            x_1 = tf.layers.dense(tf.concat([obs, action], axis=-1)  , 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

           
            x_2 = tf.layers.dense(tf.concat([obs, action], axis=-1) , 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x_1, x_2
        
            
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
    
    def TQC_call_critic(self, obs, action):

        
        
        q_target = tf.stop_gradient(self.Q_target())
        
        
        tau = ((2 * tf.range(self.num_quantiles) + 1) / (2 * self.num_quantiles))[None, None, :]


        self.q = tuple(q([obs, action]) for q in self.Q)
        
        q = tuple(tf.tile(q[:, None, :], (1, self.total_active_quantiles, 1)) for q in self.q)
        
        q_target = tf.tile(q_target[:, :, None], (1, 1, self.num_quantiles))
        all_q_loss = tuple((q_target - q_)** 2 * tf.math.abs(tau - tf.cast((q_target < q_))) for q_ in q)
        
        
        q_loss = tuple(tf.reduce_mean(l) for l in all_q_loss)

        optimisers = tuple(tf.train.AdamOptimizer(learning_rate=self._Q_lr) for i, Q in enumerate(self._Qs))

        grads = tuple(q_o.minimize(loss=q_l, var_list=q.trainable_variables)for i, (q, q_l, q_o)in enumerate(zip(q, q_loss, optimisers)))

        return grads

    def return_Q_target_TQC(self, obs, actions, reward, terminal):
        pi, log_pi = self.probability_logarithm_reparameterised([obs])
        
        

        q = tf.concat(tuple(q([obs, pi]) for q in self.Q_targets), axis=1)
        q= tf.sort(q, axis=1)
        truncated_q = q[:, :self.total_active_quantiles]
        predicted = truncated_q - self.alpha * log_pi

        
        return  td_target(reward, discount=self.gamma, next_value=(1 - terminal) * predicted)
    
    def TQC_actor_call(self, obs, action):

        pi, log_pi = self.probability_logarithm_reparameterised([obs])
        

        alpha =  tf.exp(tf.get_variable('alpha',dtype=tf.float32, initializer=0.0))
        

        factor = log_pi + self.target_entropy
        alpha_loss = -tf.reduce_mean(tf.get_variable('alpha',dtype=tf.float32, initializer=0.0) * tf.stop_gradient(factor))

        self.alpha_optimizer = tf.train.AdamOptimizer(self._Q_lr, name='alpha_optimizer') 
        self.alpha_op = self.alpha_optimizer.minimize(loss=alpha_loss, var_list=list(tf.get_variable('alpha',dtype=tf.float32)))
        self.training_op.update({'temperature_alpha': self.alpha_op})

        
        gaussian_log_prob = likelihood(obs, pi, log_pi, mode = 1) 
        targets = tf.concat(tuple(q([obs, action]) for q in self.Q), axis =1)
        
        
        targets = tf.sort(targets, axis=1)
        truncated_targets = targets[:, :self.total_active_quantiles]
        targets = tf.reduce_mean(truncated_targets, axis=1, keepdims=True)

       
        policy_loss = tf.reduce_mean(alpha * log_pi - targets - gaussian_log_prob)
        
        self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,)

        grads = self.policy_optimizer.minimize(loss=policy_loss,var_list=self.policy.trainable_variables)

        return grads
    
    
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def quantile_huber_loss_f(quantiles, target):
    pairwise_delta = target[:, None, None, :] - quantiles[:, :, :, None]  
    abs_pairwise_delta = tf.math.abs(pairwise_delta)
    huber_loss = tf.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = tf.arange(n_quantiles).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (tf.math.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

#Built upon from Mutual Information Intrinsic Control
class Off_Policy_Structure(object):
    @store_args
    def __init__(self, input_dims, buffer_size, network_width, layers, network_policy_value, network_class_ir, polyak, batch_size, Q_function_learning_rate, policy_learning_rate, mutual_information_learning_rate, skill_learning_rate, reward_scaling, mutual_information_normalization_coefficient, learn_multiple_skills, reward_scale, normalizer_small_constant, normalization_clip_value, maximum_action_value, regularisation_coefficient, clip_observations, scope, T, rollout_batch_size, subtract_goals, relative_goals ,clip_pos_returns, clip_return, sample_transitions, gamma, env_name, max_timesteps, pretrain_weights, refine_policy, mutual_information_use_in_buffer, stochastic, reuse=False, history_len=10000, **kwargs):

        
        
        dtype = tf.float32
        self.create_policy_value = import_function(self.network_policy_value)
        self.create_ir = import_function(self.network_class_ir)

        input_shapes = dims_to_shapes(self.input_dims)

        self.goal_space_dimension, self.skill_space_dimension, self.observation_space_dimension, self.action_space_dimension = self.input_dims['g'], self.input_dims['z'], self.input_dims['o'], self.input_dims['u']

        self.env_name = env_name
        
        if self.clip_return is None:
            self.clip_return = np.inf


        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['m'], stage_shapes['r'], stage_shapes['s'], stage_shapes['w'] = (None,), (None,), (None,), (None,)
        stage_shapes['r_w'], stage_shapes['m_w'], stage_shapes['e_w'], stage_shapes['s_w'] = (), (), (), ()


        self.stage_shapes = stage_shapes


        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(dtypes=[dtype for i in self.stage_shapes.keys()] , shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [tf.placeholder(dtype, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(pretrain_weights, mutual_information_use_in_buffer, reuse=reuse)



        self.et_r_history, self.gl_r_history, self.mi_r_history, self.sk_r_history  = deque(maxlen=history_len), deque(maxlen=history_len), deque(maxlen=history_len), deque(maxlen=history_len)
        self.refine_policy, self.mi_current  = refine_policy, 0
        
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key]) for key, _ in input_shapes.items()}

        buffer_shapes['g'] = (buffer_shapes['g'][0], self.goal_space_dimension)
        buffer_shapes['ag'] = (self.T+1, self.goal_space_dimension)

        self.buffer = Buffer_Structure(buffer_shapes, buffer_size, self.T, self.sample_transitions, mutual_information_use_in_buffer)


    def _random_action(self, n):
        return np.random.uniform(low=-self.maximum_action_value, high=self.maximum_action_value, size=(n, self.action_space_dimension))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            ag = ag.reshape(-1, self.goal_space_dimension)
            g = g.reshape(-1, self.goal_space_dimension)
            
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
            
        g = np.clip(g, -self.clip_observations, self.clip_observations)
        o = np.clip(o, -self.clip_observations, self.clip_observations)

        return o, g

    def get_actions(self, o, z, ag, g, noise_constant=0., small_constant_for_exploration=0., use_target_net=False, compute_Q=False):
        
        policy = self.target if use_target_net else self.main
        o, g = self._preprocess_og(o, ag, g)
        
        if self.stochastic:
            vals = [policy.mean_reparameterised]
        else:
            vals = [policy.policy_probability]
            
        if compute_Q:
            vals += [policy.Q_probability]

        feed = {policy.observation_policy_value: o.reshape(-1, self.observation_space_dimension), policy.skill_policy_value: z.reshape(-1, self.skill_space_dimension), policy.goal_policy_value: g.reshape(-1, self.goal_space_dimension),
            policy.action_policy_value: np.zeros((o.size // self.observation_space_dimension, self.action_space_dimension), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)

 
        u = ret[0]
        noise = noise_constant * self.maximum_action_value * np.random.randn(*u.shape)  
        u = u + noise
        u = np.clip(u, -self.maximum_action_value, self.maximum_action_value)
        u = u + np.random.binomial(1, small_constant_for_exploration, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u) 
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):

  
        episode_batch['m'], episode_batch['s'] = np.empty([episode_batch['o'].shape[0], 1]), np.empty([episode_batch['o'].shape[0], 1])
   
        self.buffer.add_interaction_to_replay_buffer(episode_batch, self)

        if update_stats:
           
            episode_batch['o_2'], episode_batch['ag_2'] = episode_batch['o'][:, 1:, :], episode_batch['ag'][:, 1:, :]
            shape = episode_batch['u'].shape 
            num_normalizing_transitions = shape[0] * shape[1]
            
            transitions = self.sample_transitions(self, False, episode_batch, num_normalizing_transitions, 0, 0, 0)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

            
            self.g_stats.amend(transitions['g'])
            self.o_stats.amend(transitions['o'])

            self.o_stats.recalculate()
            self.g_stats.recalculate()

    def get_current_buffer_size(self):
        return self.buffer.return_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        self.mi_adam.sync()
        self.sk_adam.sync()

    def _grads_mi(self, data):
        mi, mi_grad = self.sess.run([self.main_ir.MI_loss,self.mi_grad_tf,], feed_dict={self.o_tau_tf: data})
        return mi, mi_grad

    def _grads_sk(self, o_s_batch, z_s_batch):
        sk, sk_grad = self.sess.run([self.main_ir.skill_loss,self.sk_grad_tf,], feed_dict={self.main_ir.observation: o_s_batch, self.main_ir.skill: z_s_batch})
        return sk, sk_grad

    def _grads(self):
        critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w = self.sess.run([
            self.Q_loss_tf,self.main.Q_probability, self.Q_grad_tf,self.pi_grad_tf,
            self.main.negative_log_probability_policy,self.e_w_tf,
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w

    def _update_mi(self, mi_grad):
        self.mi_adam.update(mi_grad, self.mutual_information_learning_rate)

    def _update_sk(self, sk_grad):
        self.sk_adam.update(sk_grad, self.skill_learning_rate)

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_function_learning_rate)
        self.pi_adam.update(pi_grad, self.policy_learning_rate)

    def sample_batch(self, ir, t):

        transitions = self.buffer.return_trajectories(self, ir, self.batch_size, self.mutual_information_normalization_coefficient, self.learn_multiple_skills, t)
        weights = np.ones_like(transitions['r']).copy()
        if ir:
            self.sk_r_history.extend(((np.clip(self.learn_multiple_skills * transitions['s'], *(-1, 0)))*1.00).tolist())
            self.mi_r_history.extend(((np.clip((self.mutual_information_normalization_coefficient * transitions['m']), *(0, 1))- (1 if not self.mutual_information_normalization_coefficient == 0 else 0) )*transitions['m_w']).tolist())
            self.gl_r_history.extend(self.reward_scaling * transitions['r'])

        o = transitions['o']
        o_2 = transitions['o_2']
        g = transitions['g']
        ag = transitions['ag']
        ag_2 =  transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions['w'] = weights.flatten().copy() 
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch

    def stage_batch(self, ir, t, batch=None):
        if batch is None:
            batch = self.sample_batch(ir, t)
        assert len(self.buffer_ph_tf) == len(batch) # 
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def run_mi(self, o_s):
        
        neg_l = self.sess.run(self.main_ir.MI_loss, feed_dict={self.o_tau_tf: o_s.copy()})
        return neg_l

    def run_sk(self, o, z):
        
        sk_r = self.sess.run(self.main_ir.skill_reward, feed_dict={self.main_ir.observation: o, self.main_ir.skill: z})
        return sk_r

    def train_mi(self, data, stage=True):
        mi, mi_grad = self._grads_mi(data)
        self._update_mi(mi_grad)
        self.mi_current = -mi.mean()
        return -mi.mean()

    def train_sk(self, o_s_batch, z_s_batch, stage=True):
        sk, sk_grad = self._grads_sk(o_s_batch, z_s_batch)
        self._update_sk(sk_grad)
        return -sk.mean()

    def train(self, t, stage=True):
        if not self.buffer.cur_capacity==0:
            if stage:
                self.stage_batch(ir=True, t=t)
            critic_loss, actor_loss, Q_grad, pi_grad, neg_logp_pi, e_w = self._grads()
            self._update(Q_grad, pi_grad)
            self.et_r_history.extend((( np.clip((self.reward_scale * neg_logp_pi), *(-1, 0))) * e_w ).tolist())
            return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.reset_replay_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0 # 
        return res

    def _global_vars(self, scope):
        
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)

    def _create_network(self, pretrain_weights, mutual_information_use_in_buffer, reuse=False):
   
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Clip_And_Standardise(self.observation_space_dimension, self.normalizer_small_constant, self.normalization_clip_value, session=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Clip_And_Standardise(self.goal_space_dimension, self.normalizer_small_constant, self.normalization_clip_value, session=self.sess)


        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(val, batch[j]) for j, val in enumerate(self.stage_shapes.keys())])
        batch_tf['m'], batch_tf['r'], batch_tf['w'], batch_tf['s'] = tf.reshape(batch_tf['m'], [-1, 1]), tf.reshape(batch_tf['r'], [-1, 1]), tf.reshape(batch_tf['w'], [-1, 1]), tf.reshape(batch_tf['s'], [-1, 1])

        self.o_tau_tf = tf.placeholder(tf.float32, shape=(None, None, self.observation_space_dimension))
        e_mode, mode = 1, 1
        if e_mode == 1:
            with tf.variable_scope('main') as vs:
                if reuse:
                    vs.reuse_variables()
                self.main = self.create_policy_value(batch_tf, net_type='main', **self.__dict__)
                vs.reuse_variables()
        elif e_mode == 2:
             with tf.variable_scope('main_1') as vs:
               if reuse:
                    vs.reuse_variables()
               self.main_1 = self.create_policy_value(batch_tf, net_type='main', name_is = 'main_1', **self.__dict__)
             with tf.variable_scope('main_2') as vs:
               if reuse:
                    vs.reuse_variables()
               self.main_2 = self.create_policy_value(batch_tf, net_type='main', name_is = 'main_2', **self.__dict__)
             with tf.variable_scope('main_3') as vs:
               if reuse:
                    vs.reuse_variables()
               self.main_3 = self.create_policy_value(batch_tf, net_type='main', name_is = 'main_3', **self.__dict__)
             with tf.variable_scope('main_4') as vs:
               if reuse:
                    vs.reuse_variables()
               self.main_4 = self.create_policy_value(batch_tf, net_type='main', name_is = 'main_4', **self.__dict__)
            
               #self.main_2 = self.create_policy_value(batch_tf, net_type='main', name_is = 'main_2', **self.__dict__)
               #self.main_3 = self.create_policy_value(batch_tf, net_type='main', name_is = 'main_3', **self.__dict__)
               #self.main_4 = self.create_policy_value(batch_tf, net_type='main', name_is = 'main_4', **self.__dict__)
            
               
              
        
        if e_mode == 1:  
            with tf.variable_scope('target') as vs:
                if reuse:
                    vs.reuse_variables()
                target_batch_tf = batch_tf.copy()
                target_batch_tf['o'] = batch_tf['o_2']
                target_batch_tf['g'] = batch_tf['g_2']
                self.target = self.create_policy_value(target_batch_tf, net_type='target', **self.__dict__)
                vs.reuse_variables()
        elif e_mode == 2:
            with tf.variable_scope('target_1') as vs:
                if reuse:
                    vs.reuse_variables()
                target_batch_tf = batch_tf.copy()
                target_batch_tf['o'] = batch_tf['o_2']
                target_batch_tf['g'] = batch_tf['g_2']
                self.target_1 = self.create_policy_value(target_batch_tf, net_type='target', name_is = 'target_1', **self.__dict__)
                vs.reuse_variables()
            with tf.variable_scope('target_2') as vs:
                if reuse:
                    vs.reuse_variables()
                target_batch_tf = batch_tf.copy()
                target_batch_tf['o'] = batch_tf['o_2']
                target_batch_tf['g'] = batch_tf['g_2']
                self.target_2 = self.create_policy_value(target_batch_tf, net_type='target', name_is = 'target_2', **self.__dict__)
                vs.reuse_variables()
            
            with tf.variable_scope('target_3') as vs:
                if reuse:
                    vs.reuse_variables()
                target_batch_tf = batch_tf.copy()
                target_batch_tf['o'] = batch_tf['o_2']
                target_batch_tf['g'] = batch_tf['g_2']
                self.target_3 = self.create_policy_value(target_batch_tf, net_type='target', name_is = 'target_3', **self.__dict__)
                vs.reuse_variables()
            with tf.variable_scope('target_4') as vs:
                if reuse:
                    vs.reuse_variables()
                target_batch_tf = batch_tf.copy()
                target_batch_tf['o'] = batch_tf['o_2']
                target_batch_tf['g'] = batch_tf['g_2']
                self.target_4 = self.create_policy_value(target_batch_tf, net_type='target', name_is = 'target_4', **self.__dict__)
            
                vs.reuse_variables()
            
        assert len(self._vars("main")) == len(self._vars("target")) # 


        with tf.variable_scope('ir') as vs:
            if reuse:
                vs.reuse_variables()
            self.main_ir = self.create_ir(batch_tf, net_type='ir', **self.__dict__)
            vs.reuse_variables()

      

        mi_grads_tf = tf.gradients(tf.reduce_mean(self.main_ir.MI_loss), self._vars('ir/state_mi'))
        assert len(self._vars('ir/state_mi')) == len(mi_grads_tf)
        self.mi_grads_vars_tf = zip(mi_grads_tf, self._vars('ir/state_mi'))
        self.mi_grad_tf = flatten_grads(grads=mi_grads_tf, var_list=self._vars('ir/state_mi'))
        self.mi_adam = MpiAdam(self._vars('ir/state_mi'), scale_grad_by_procs=False)

        sk_grads_tf = tf.gradients(tf.reduce_mean(self.main_ir.skill_loss), self._vars('ir/skill_ds'))
        assert len(self._vars('ir/skill_ds')) == len(sk_grads_tf)
        self.sk_grads_vars_tf = zip(sk_grads_tf, self._vars('ir/skill_ds'))
        self.sk_grad_tf = flatten_grads(grads=sk_grads_tf, var_list=self._vars('ir/skill_ds'))
        self.sk_adam = MpiAdam(self._vars('ir/skill_ds'), scale_grad_by_procs=False)

        if e_mode == 1:
            target_Q_pi_tf = self.target.Q_probability
        elif e_mode == 2: 
          target_Q_pi_tf_1 = self.target_1.Q_probability
          target_Q_pi_tf_2 = self.target_2.Q_probability
          target_Q_pi_tf_3 = self.target_3.Q_probability
          target_Q_pi_tf_4 = self.target_4.Q_probability
          list_of_Q_values = [target_Q_pi_tf, target_Q_pi_tf_1, target_Q_pi_tf_2, target_Q_pi_tf_3, target_Q_pi_tf_4]   
            
        clip_range = (-self.clip_return, self.clip_return if self.clip_pos_returns else np.inf)

        self.e_w_tf = batch_tf['e_w']
        
        
        if mode == 2:

           reshaped = tf.reshape(target_Q_pi_tf, (self.batch_size, -1))
           reshaped = reshaped.eval(session=self.sess)   
           sorted_z, _ = tf.sort(reshaped)
           target_Q_pi_tf = sorted_z[:, :125-10]


        if not self.stochastic:
            self.main.negative_log_probability_policy = tf.zeros(1)

        target_tf = tf.clip_by_value(self.reward_scaling * batch_tf['r'] * batch_tf['r_w'] + (tf.clip_by_value( self.mutual_information_normalization_coefficient * batch_tf['m'], *(0, 1) ) - (1 if not self.mutual_information_normalization_coefficient == 0 else 0)) * batch_tf['m_w'] + (tf.clip_by_value( self.learn_multiple_skills * batch_tf['s'], *(-1, 0))) * batch_tf['s_w'] + (tf.clip_by_value( self.reward_scale * self.main.negative_log_probability_policy, *(-1, 0))) * self.e_w_tf + self.gamma * target_Q_pi_tf, *clip_range)
        
        if mode == 2: 
            self.Q_loss_tf = quantile_huber_loss_f(self.main.Q_values_policy_value, target = tf.stop_gradient(target_tf))
        else:
            self.td_error_tf = tf.stop_gradient(target_tf) - self.main.Q_values_policy_value
            self.errors_tf = tf.square(self.td_error_tf)
            self.errors_tf = tf.reduce_mean(batch_tf['w'] * self.errors_tf)
            self.Q_loss_tf = tf.reduce_mean(self.errors_tf)
            
        if e_mode == 2:
             std_Q = statistics.stdev(list_of_Q_values)
             temperature = 0.1 
             weight = sigmoid(temperature * std_Q) + 0.5
             self.Q_loss_tf = self.Q_loss_tf * weight

        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_probability)
        self.pi_loss_tf += self.regularisation_coefficient * tf.reduce_mean(tf.square(self.main.policy_probability / self.maximum_action_value))
        
        if e_mode == 2:
            pi_loss_1 = self.e_w_tf * -self.main_1.neg_logp_pi_tf - tf.reduce_mean(self.main_1.Q_probability)
            pi_loss_2 = self.e_w_tf * -self.main_2.neg_logp_pi_tf - tf.reduce_mean(self.main_2.Q_probability)
            pi_loss_3 = self.e_w_tf * -self.main_3.neg_logp_pi_tf - tf.reduce_mean(self.main_3.Q_probability)
            pi_loss_4 = self.e_w_tf * -self.main_4.neg_logp_pi_tf - tf.reduce_mean(self.main_4.Q_probability)
    
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        
        
        
        if e_mode == 2:
            Q_grads_tf_1 = tf.gradients(self.Q_loss_tf, self._vars('main_1/Q'))
            Q_grads_tf_2 = tf.gradients(self.Q_loss_tf, self._vars('main_2/Q'))
            Q_grads_tf_3 = tf.gradients(self.Q_loss_tf, self._vars('main_3/Q'))
            Q_grads_tf_4 = tf.gradients(self.Q_loss_tf, self._vars('main_4/Q'))
            pi_grads_tf_1 = tf.gradients(self.pi_loss_1, self._vars('main_1/pi'))
            pi_grads_tf_2 = tf.gradients(self.pi_loss_2, self._vars('main_2/pi'))
            pi_grads_tf_3 = tf.gradients(self.pi_loss_3, self._vars('main_3/pi'))
            pi_grads_tf_4 = tf.gradients(self.pi_loss_4, self._vars('main_4/pi'))
        
        assert len(self._vars('main/Q')) == len(Q_grads_tf) # 
        assert len(self._vars('main/pi')) == len(pi_grads_tf) # 
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        if e_mode == 2:
            self.Q_grads_vars_tf_1 = zip(Q_grads_tf_1, self._vars('main_1/Q'))
            self.pi_grads_vars_tf_1 = zip(pi_grads_tf_1, self._vars('main_1/pi'))
            self.Q_grad_tf_1 = flatten_grads(grads=Q_grads_tf_1, var_list=self._vars('main_1/Q'))
            self.pi_grad_tf_1 = flatten_grads(grads=pi_grads_tf_1, var_list=self._vars('main_1/pi'))
            self.Q_grads_vars_tf_2 = zip(Q_grads_tf_2, self._vars('main_2/Q'))
            self.pi_grads_vars_tf_2 = zip(pi_grads_tf_2, self._vars('main_2/pi'))
            self.Q_grad_tf_2 = flatten_grads(grads=Q_grads_tf_2, var_list=self._vars('main_2/Q'))
            self.pi_grad_tf_2 = flatten_grads(grads=pi_grads_tf_2, var_list=self._vars('main_2/pi'))
            self.Q_grads_vars_tf_3 = zip(Q_grads_tf_3, self._vars('main_3/Q'))
            self.pi_grads_vars_tf_3 = zip(pi_grads_tf_3, self._vars('main_3/pi'))
            self.Q_grad_tf_3 = flatten_grads(grads=Q_grads_tf_3, var_list=self._vars('main_3/Q'))
            self.pi_grad_tf_3 = flatten_grads(grads=pi_grads_tf_3, var_list=self._vars('main_3/pi'))
            self.Q_grads_vars_tf_4 = zip(Q_grads_tf_4, self._vars('main_4/Q'))
            self.pi_grads_vars_tf_4 = zip(pi_grads_tf_4, self._vars('main_4/pi'))
            self.Q_grad_tf_4 = flatten_grads(grads=Q_grads_tf_4, var_list=self._vars('main_4/Q'))
            self.pi_grad_tf_4 = flatten_grads(grads=pi_grads_tf_4, var_list=self._vars('main_4/pi'))

        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)
        
        if e_mode == 2:
            self.Q_adam_1 = MpiAdam(self._vars('main_1/Q'), scale_grad_by_procs=False)
            self.pi_adam_1 = MpiAdam(self._vars('main_1/pi'), scale_grad_by_procs=False)
            self.Q_adam_2 = MpiAdam(self._vars('main_2/Q'), scale_grad_by_procs=False)
            self.pi_adam_2 = MpiAdam(self._vars('main_2/pi'), scale_grad_by_procs=False)
            self.Q_adam_3 = MpiAdam(self._vars('main_3/Q'), scale_grad_by_procs=False)
            self.pi_adam_3 = MpiAdam(self._vars('main_3/pi'), scale_grad_by_procs=False)
            self.Q_adam_4 = MpiAdam(self._vars('main_4/Q'), scale_grad_by_procs=False)
            self.pi_adam_4 = MpiAdam(self._vars('main_4/pi'), scale_grad_by_procs=False)

        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        
        if e_mode == 2:
            self.main_vars_1 = self._vars('main_1/Q') + self._vars('main_1/pi')
            self.target_vars_1 = self._vars('target_1/Q') + self._vars('target_1/pi')
            self.main_vars_2 = self._vars('main_2/Q') + self._vars('main_2/pi')
            self.target_vars_2 = self._vars('target_2/Q') + self._vars('target_2/pi')
            self.main_vars_3 = self._vars('main_3/Q') + self._vars('main_3/pi')
            self.target_vars_3 = self._vars('target_3/Q') + self._vars('target_3/pi')
            self.main_vars_4 = self._vars('main_4/Q') + self._vars('main_4/pi')
            self.target_vars_4 = self._vars('target_4/Q') + self._vars('target_4/pi')


        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        if e_mode == 2: 
            self.init_target_net_op_1 = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars_1, self.main_vars_1)))
            self.update_target_net_op_1 = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars_1, self.main_1)))
            self.init_target_net_op_2 = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars_2, self.main_vars_2)))
            self.update_target_net_op_2 = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars_2, self.main_2)))
            self.init_target_net_op_3 = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars_3, self.main_vars_3)))
            self.update_target_net_op_3 = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars_3, self.main_3)))
            self.init_target_net_op_4 = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars_4, self.main_vars_4)))
            self.update_target_net_op_4 = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars_4, self.main_4)))

        tf.variables_initializer(self._global_vars('')).run()
        if pretrain_weights:
            load_weight(self.sess, pretrain_weights, ['state_mi']) 
            if self.refine_policy:
                load_weight(self.sess, pretrain_weights, ['main'])

        self._sync_optimizers()
        if pretrain_weights and self.refine_policy:
            load_weight(self.sess, pretrain_weights, ['target'])
        else:
            self._init_target_net()
    
  

    def logs(self, prefix=''):
        logs = []
        logs = logs + [('stats_o/mean', np.mean(self.sess.run([self.o_stats.avg_var])))]
        logs = logs + [('stats_o/std', np.mean(self.sess.run([self.o_stats.sigma])))]
        logs = logs + [('stats_g/mean', np.mean(self.sess.run([self.g_stats.avg_var])))]
        logs = logs + [('stats_g/std', np.mean(self.sess.run([self.g_stats.sigma])))]
        logs = logs + [('mi_reward/mean', np.mean(self.mi_r_history))]
        logs = logs + [('mi_reward/std', np.std(self.mi_r_history))]
        logs = logs + [('mi_reward/max', np.max(self.mi_r_history))]
        logs = logs + [('mi_reward/min', np.min(self.mi_r_history))]
        logs = logs + [('mi_train/-neg_l', self.mi_current)]
        logs = logs + [('gl_reward/mean', np.mean(self.gl_r_history))]
        logs = logs + [('gl_reward/std', np.std(self.gl_r_history))]
        logs = logs + [('gl_reward/max', np.max(self.gl_r_history))]
        logs = logs + [('gl_reward/min', np.min(self.gl_r_history))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):

        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'sample_transitions',
                             'stage_shapes', 'create_policy_value', 'create_ir', '_history']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            state['sample_transitions'] = None
        if 'env_name' not in state:
            state['env_name'] = 'FetchPickAndPlace-v1'
        if 'network_class_ir' not in state:
            state['network_class_ir'] = 'intrinsic_rewards_estimator:Estimator_Of_Intrinsic_Rewards'
        if 'mutual_information_normalization_coefficient' not in state:
            state['mutual_information_normalization_coefficient'] = 1
        if 'mutual_information_learning_rate' not in state:
            state['mutual_information_learning_rate'] = 0.001
        if 'learn_multiple_skills' not in state:
            state['learn_multiple_skills'] = 1
        if 'skill_learning_rate' not in state:
            state['skill_learning_rate'] = 0.001
        if 'reward_scale' not in state:
            state['reward_scale'] = 1
        if 'refine_policy' not in state:
            state['refine_policy'] = None
        if 'negative_train_of_mutual_information' not in state:
            state['negative_train_of_mutual_information'] = None
        if 'load_weight' not in state:
            state['load_weight'] = None
        if 'pretrain_weights' not in state:
            state['pretrain_weights'] = None
        if 'mutual_information_use_in_buffer' not in state:
            state['mutual_information_use_in_buffer'] = None
        if 'stochastic' not in state:
            state['stochastic'] = None

        self.__init__(**state)

        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v

        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    
