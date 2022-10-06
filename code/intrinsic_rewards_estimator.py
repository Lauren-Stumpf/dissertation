



import tensorflow as tf

from general_util import store_args, nn 
import tensorflow.contrib.layers as tf_layers


#Adapted from Mutual Information State Intrinsic Control 
class Estimator_Of_Intrinsic_Rewards:
    @store_args
    def __init__(self, interaction_inp, observation_space_dimension, skill_space_dimension, goal_space_dimension, action_space_dimension, maximum_action_value, o_stats, g_stats, network_width, layers, env_name, **kwargs):


        self.observation = tf.placeholder(tf.float32, shape=(None, self.observation_space_dimension))
        self.skill = tf.placeholder(tf.float32, shape=(None, self.skill_space_dimension))
        self.goal = tf.placeholder(tf.float32, shape=(None, self.goal_space_dimension))

        incomplete_observation, complete_observation, _, _, _, _, _, _, _ = tf.split(self.o_tau_tf, [3, 3, 3, 2, 3, 3, 3, 3, 2], axis=-1)
        observation_without_goal_pos, observation_with_goal_pos, _, _, _, _, _, _, _ = tf.split(self.observation, [3, 3, 3, 2, 3, 3, 3, 3, 2], axis=-1)

        
        with tf.variable_scope('state_mi'):


            agent_state = tf.transpose(complete_observation, perm=[1, 0, 2])
            agent_state_shuffled = tf.random_shuffle(agent_state)
            
            conc_agent_state = tf.concat([complete_observation, tf.transpose(agent_state_shuffled, perm=[1, 0, 2])], axis=-2)
            agent_layer = tf_layers.linear(conc_agent_state, int(self.network_width/2))
            
            conc_env_state = tf.concat([incomplete_observation, incomplete_observation], axis=-2)
            env_layer = tf_layers.linear(conc_env_state, int(self.network_width/2))
            
            agent_layer = tf.nn.relu(env_layer + agent_layer)
            output = tf.nn.tanh(tf_layers.linear(agent_layer, 1))
      

            self.MI_loss = _loss_(output[:,:tf.shape(incomplete_observation)[-2],:], output[:,tf.shape(incomplete_observation)[-2]:,:])

        with tf.variable_scope('skill_ds'):
            self.log_values = nn(observation_with_goal_pos, [int(self.network_width/2)] * self.layers + [self.skill_space_dimension])
            self.skill_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.skill, logits=self.log_values)
            self.skill_reward = -1 * self.skill_loss
            
def _loss_(X, Y):
    e = tf.reduce_mean(tf.math.exp(Y), axis=-2)
    loss = -(tf.reduce_mean(X, axis=-2) - tf.math.log(e))
  
    return loss



import numpy as np

#my implementation of paper Mutual Information Neural Estimation https://arxiv.org/abs/1801.04062
def wrapper_for_tf_shuffle(y):
    return tf.random_shuffle(y)



def MI_loss(losses, mode):
    if mode == 1:
        loss = -(tf.math.reduce_mean(losses[0]) - tf.math.reduce(losses[1]) + tf.math.log(tf.cast(losses[0][0], tf.float32)))
    elif mode == 2:
        
        loss = (1. - tf.math.sigmoid(losses[0])) * losses[0] - logsigmoid(losses[1])
        

    return loss


class Mutual_Information_Neural_Network(object):
    def __init__(self, size_x, size_y):
        self.model, self.size_x, self.size_y = None, size_x, size_y
        
        self.estimator = self.construct_MINE()
        
    def construct_MINE(self):
        
        y = tf.keras.Input(shape=(self.size_y,))
        x = tf.keras.Input(shape=(self.size_x,))
        
        con = tf.keras.layers.Concatenate([x, y])
        
        hidden = tf.layers.Dense(256, activation='relu')(con)
        
        hidden_1 = tf.layers.Dense(256, activation='relu')(hidden)
        
        hidden_2 = tf.layers.Dense(256, activation='relu')(hidden_1)
        
        final_layer = tf.layers.Dense(1)(hidden_2)
        
        model = tf.keras.Model(inputs=[x, y], outputs=final_layer)
        return model

    def predict(self, x, y):
        return np.mean(self.model.predict(x=(x + y)))
    
    def fit(self, x, y, epochs, batch_size, verbose):
        self.make_MINE()
        
        history = self.model.fit(x=(x+y), epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        loss = history.history['loss']
        
        MI = self.predict(x, y)
        
        return loss, MI

    def make_MINE(self):
       
        y_sample = tf.keras.layers.Lambda(wrapper_for_tf_shuffle)(self.estimator.inputs[-1] )  
        
        x_y = self.estimator(self.estimator.inputs[0:-1] + [self.estimator.inputs[-1] ])
        
        x_y_ = self.estimator(self.estimator.inputs[0:-1] + [y_sample])
        
        loss = tf.keras.layers.Lambda(MI_loss)([x_y, x_y_])
        
        output = tf.keras.layers.Lambda(lambda x: -x)(loss)
        
        self.model = tf.keras.Model(inputs=self.estimator.inputs[0:-1] + [self.estimator.inputs[-1] ], outputs=output, name='MINE_model')
        
        self.model.add_loss(loss)
        
        self.model.compile(optimizer='adam')
        

    def reward(self, x):
        out = self(x)
        probs = tf.math.sigmoid(out)
        return tf.math.log(probs + 1e-8) - tf.math.log(1 - probs + 1e-8)



    def ce_loss(self, pred_on_learner, pred_on_expert, expert_weights):
        
        learner_loss = tf.math.log(1 - tf.math.sigmoid(pred_on_learner))
        expert_loss = tf.math.log(tf.math.sigmoid(pred_on_expert)) * expert_weights
        return -self.agg(learner_loss + expert_loss)



    def gradient_penalty(self, mi_sample, mi_predicted):
        
        batch_size = mi_sample.size()[0]

        alpha = tf.random.uniform(batch_size, 1)
        alpha = alpha.expand_as(mi_sample)
        
        interpolated = alpha * mi_sample.data + (1 - alpha) * mi_predicted.data
        
        gradients = tf.grad(outputs=mi_sample, inputs=mi_predicted,grad_outputs=tf.ones(interpolated.size()))[0]

    
        gradients = gradients.view(batch_size, -1)

        return self.LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
def logsigmoid(x):
    return tf.math.log(tf.math.sigmoid(x))

            
