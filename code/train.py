

### TRAIN 
# Everything is adapted from Mutual Information State Intrinsic Control
import os
import sys
import pickle

import numpy as np
import json

import gym



import logger
from agent import Off_Policy_Structure
from experience_replay import her_function
import click

from mpi4py import MPI


from general_util import set_global_seeds, dumpJson, save_video
from MPI_util import mpi_moments

from distribute import Distribute_Workers_Class
from general_util import mpi_fork

import os.path as osp
import tempfile
import datetime





import tensorflow as tf
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


CACHED_ENVS = {}
def cached_make_env(make_env):

    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):

    Off_Policy_Structure_params = dict()

    env_name = kwargs['env_name']
    def make_env():
        return gym.make(env_name)
    
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    
 
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    
    
    if 'lr' in kwargs:
        kwargs['policy_learning_rate'] = kwargs['lr']
        kwargs['Q_function_learning_rate'] = kwargs['lr']
        del kwargs['lr']
    
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    kwargs['maximum_action_value'] = np.array(kwargs['maximum_action_value']) if type(kwargs['maximum_action_value']) == list else kwargs['maximum_action_value']

    
    for val in ['buffer_size','polyak','normalizer_small_constant','policy_learning_rate','scope',
                    'relative_goals','normalization_clip_value','network_width','network_policy_value',
                    'layers','skill_learning_rate','maximum_action_value','regularisation_coefficient','clip_observations',
                    'network_class_ir','Q_function_learning_rate', 'batch_size', 'mutual_information_learning_rate']:
        Off_Policy_Structure_params[val] = kwargs[val]
        kwargs['_' + val] = kwargs[val]
        del kwargs[val]
        
    kwargs['Off_Policy_Structure_params'] = Off_Policy_Structure_params

    return kwargs



def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    
    
    def return_reward(ag_2, g, info): 
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    her_params = {'return_reward': return_reward,}
    for p in ['mutual_information_use_in_buffer', 'schedule_for_mutual_information','her_replay_parameter',  'her_buffer_strategy', 'weight_scheduler']:
        her_params[p] = params[p]
        if not (p in ['mutual_information_use_in_buffer']):
            params['_' + p] = her_params[p]
            del params[p]

    sample_her_transitions = her_function(**her_params)

    return sample_her_transitions


def simple_goal_subtract(a, b):
   
    return a - b


def configure_Off_Policy_Structure(dims, params, pretrain_weights, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    
    Off_Policy_Structure_params = params['Off_Policy_Structure_params']
    

    input_dims = dims.copy()

    
    env = cached_make_env(params['make_env'])
    env.reset()
    Off_Policy_Structure_params.update({'input_dims': input_dims, 
                        'T': params['T'], 'clip_pos_returns': True,  
                        'clip_return': (1. / (1. - params['gamma']))*params['input_dimensions'] if clip_return else np.inf,  
                        'rollout_batch_size': params['rollout_batch_size'],'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,'gamma': params['gamma'],'env_name': params['env_name'],
                        'max_timesteps': params['max_timesteps'],'reward_scaling': params['reward_scaling'], 'mutual_information_normalization_coefficient': params['mutual_information_normalization_coefficient'],
                        'maximum_number_of_epochs': params['maximum_number_of_epochs'],'learn_multiple_skills': params['learn_multiple_skills'],
                        'reward_scale': params['reward_scale'],'pretrain_weights': pretrain_weights,
                        'refine_policy': params['refine_policy'], 'mutual_information_use_in_buffer': params['mutual_information_use_in_buffer'],
                        'stochastic': params['stochastic'],
                        })
    Off_Policy_Structure_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = Off_Policy_Structure(reuse=reuse, **Off_Policy_Structure_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {'o': obs['observation'].shape[0],'z': params['num_skills'],
        'u': env.action_space.shape[0], 'g': obs['desired_goal'].shape[0],}
    
    for val, item in info.items():
        item = np.array(item)
        if item.ndim == 0:
            item = item.reshape(1)
        dims['info_{}'.format(val)] = item.shape[0]
    return dims

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def sample_skill(num_skills, rollout_batch_size, refine_multiple_skills=None):

    z_s = np.random.randint(0, num_skills, rollout_batch_size)
    if refine_multiple_skills:

        refine_multiple_skills = refine_multiple_skills - 1
        z_s.fill(refine_multiple_skills)
        
    z_s_onehot = np.zeros([rollout_batch_size, num_skills])
    z_s = np.array(z_s).reshape(rollout_batch_size, 1)
    for i in range(rollout_batch_size):
        z_s_onehot[i, z_s[i]] = 1
    return z_s, z_s_onehot

def train(logdir, policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval, save_policies, num_cpu, collect_data, collect_video, number_of_goals_to_generate_in_buffer, num_skills, refine_multiple_skills, batch_size, mutual_information_normalization_coefficient, maximum_number_of_epochs, learn_multiple_skills, negative_train_of_mutual_information, **kwargs):

    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    t = 1
    for epoch in range(n_epochs):

        # train
        episodes = []
        rollout_worker.reset_trajectories()
        for cycle in range(n_cycles):

            z_s, z_s_onehot = sample_skill(num_skills, rollout_worker.rollout_batch_size, refine_multiple_skills)

            if number_of_goals_to_generate_in_buffer == 'Zero':
                generated_goal = np.zeros(rollout_worker.goal_rollout.shape)
            else:
    
                generated_goal = False

            if collect_video: # 
                imgs, episode = rollout_worker.set_up_workers(generated_goal=generated_goal, z_s_onehot=z_s_onehot)

                for rollout in range(rollout_worker.rollout_batch_size):
                    filename = logdir + '/video_epoch_{0}_rank_{1}_cycle_{2}_rollout_{3}_skill_{4}.avi'.format(epoch, rank, cycle, rollout, z_s[rollout][0])
                    if rank == 0:
                        save_video(imgs[rollout], filename)
           
            else:
                episode = rollout_worker.set_up_workers(generated_goal=generated_goal, z_s_onehot=z_s_onehot)
            episodes.append(episode)
            policy.store_episode(episode)
            for batch in range(n_batches):
                t = epoch
                policy.train(t)


                if epoch >= maximum_number_of_epochs:
                    mutual_information_normalization_coefficient = 0
                    policy.mutual_information_normalization_coefficient = 0

                if mutual_information_normalization_coefficient > 0 and (not negative_train_of_mutual_information):
                    o_s = policy.buffer.storage_structure['o'][0: policy.buffer.cur_capacity]
                    episode_idxs = np.random.randint(0, policy.buffer.cur_capacity, batch_size)
                    o_s_batch = o_s[episode_idxs]
                    policy.train_mi(o_s_batch)
                

                

            policy.update_target_net()

        if collect_data and (rank == 0):
            dumpJson(logdir, episodes, epoch, rank)

        
        evaluator.reset_trajectories()
        for _ in range(n_test_rollouts):
            z_s, z_s_onehot = sample_skill(num_skills, evaluator.rollout_batch_size, refine_multiple_skills)
            evaluator.set_up_workers(generated_goal=False, z_s_onehot=z_s_onehot)

        
        logger.record_tabular('timestamp', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        logger.record_tabular('best_success_rate', best_success_rate) 

        if rank == 0:
            logger.dump_tabular()

        success_rate = mpi_average(evaluator.success_across_all_workers())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        


def launch(location, parameters, override_params={}, save_policies=True):
    
    params = {}

    with open(parameters+'.json', 'r') as file:
        params = json.loads(file.read())
        params.update(**override_params)

    n_epochs = params['n_epochs']
    num_cpu = params['num_cpu']
    binding = params['binding']
    logging = params['logging']
    env_name = params['env_name']
    clip_return = params['clip_return']
    policy_save_interval = params['policy_save_interval']


    if num_cpu > 1:
        whoami = mpi_fork(num_cpu, binding)
        if whoami == 'parent':
            sys.exit(0)
        import tensorflow_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()


    if logging: 
        logdir = 'logs/'+str(env_name)+'-T-'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        logdir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))

    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure() 
        
    logdir = logger.get_dir()
    
    os.makedirs(logdir, exist_ok=True)

    rank_seed =  1000000 * rank
    set_global_seeds(rank_seed)

    
    params['max_timesteps'] = n_epochs * params['n_cycles'] *  params['n_batches'] * num_cpu
    

    if location != None:
        base = os.path.splitext(location)[0]
        policy_weight_file = open(base+'_weight.pkl', 'rb')
        pretrain_weights = pickle.load(policy_weight_file)
        policy_weight_file.close()
    else:
        pretrain_weights = None


    
    params = prepare_params(params)
    #log_params(params, logger=logger)

    dims = configure_dims(params)
    policy = configure_Off_Policy_Structure(dims=dims, params=params, pretrain_weights=pretrain_weights, clip_return=clip_return)

    render = False
    if params['collect_video']:
        render = 'rgb_array'

 
    rollout_params = {'optimise': False,'double_Q_learning': False,'use_demo_states': True,
        'return_Q_values_reg': False, 'T': params['T'], 'render': render,}

    eval_params = {'optimise': True,'double_Q_learning': params['test_with_polyak'],
        'use_demo_states': False,'return_Q_values_reg': True, 'T': params['T'],}

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_constant', 'small_constant_for_exploration']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = Distribute_Workers_Class(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = Distribute_Workers_Class(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'], n_cycles=params['n_cycles'], n_batches=params['n_batches'], policy_save_interval=policy_save_interval, save_policies=save_policies, num_cpu=num_cpu, collect_data=params['collect_data'], collect_video=params['collect_video'], number_of_goals_to_generate_in_buffer=params['number_of_goals_to_generate_in_buffer'], num_skills=params['num_skills'], refine_multiple_skills=params['refine_multiple_skills'], batch_size=params['_batch_size'], mutual_information_normalization_coefficient=params['mutual_information_normalization_coefficient'], maximum_number_of_epochs=params['maximum_number_of_epochs'], learn_multiple_skills=params['learn_multiple_skills'], negative_train_of_mutual_information=params['negative_train_of_mutual_information'])


@click.command()
@click.option('--location', type=str, default=None, help='The location of the pretrained Mutual Information Neural Network')
@click.option('--parameters', type=str, default=None, help='The specific parameter created by main.py')


def main(**kwargs):
    launch(**kwargs)

if __name__ == '__main__':
    main()

