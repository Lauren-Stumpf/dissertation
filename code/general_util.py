### GENERAL UTIL

import subprocess
import sys
import numpy as np
import importlib
import random
import inspect
import functools
import os
import platform
import json
import tensorflow as tf
from tensorflow_util import numel 



def zipsame(*seqs):

    return zip(*seqs)



def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)
    
    


def linear_interpolation(starting_value, end_value, increment):
    return starting_value + increment * (end_value - starting_value)


class LinearSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):

  
      
        self.top_value = outside_value
        self.interpolation_function = interpolation
        
        self.upper_and_lower_value = endpoints

    def value(self, time):

        for (start_time, start_increment), (end_time, end_increment) in zip(self.upper_and_lower_value[:-1], self.upper_and_lower_value[1:]):
            if start_time <= time and time < end_time:
                
                interpolation_increment = float(time - start_time) / (end_time - start_time)
                
                return self.interpolation_function(start_increment, end_increment, interpolation_increment)

        return self.top_value



class _LRScheduler(object):
    def __init__(self, optimizer, final_ep=-1):
    
        self.opt = optimizer
        if final_ep == -1:
            for collection in optimizer.param_collections:
                collection.setdefault('initial_lr', collection['lr'])
        else:
            for i, collection in enumerate(optimizer.param_collections):
                if 'initial_lr' not in collection:
                    raise KeyError
        self.first_lr = list(map(lambda collection: collection['initial_lr'], optimizer.param_collections))
        self.increment(final_ep + 1)
        self.final_ep = final_ep

    def get_lr(self):
        raise NotImplementedError

    def increment(self, epoch=None):
        if epoch is None:
            epoch = self.final_ep + 1
        self.final_ep = epoch
        for param_collection, lr in zip(self.opt.param_collections, self.get_lr()):
            param_collection['lr'] = lr


class LambdaLR(_LRScheduler):

    def __init__(self, optimizer, lr_lambda, final_ep=-1):
        self.opt = optimizer
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_collections)
        else:
            if len(lr_lambda) != len(optimizer.param_collections):
                raise ValueError
                
        self.final_ep = final_ep
        super(LambdaLR, self).__init__(optimizer, final_ep)

    def get_lr(self):
        return [base_lr * lmbda(self.final_ep)
                for lmbda, base_lr in zip(self.lr_lambdas, self.first_lr)]



class incrementLR(_LRScheduler):


    def __init__(self, optimizer, increment_size, discount_factor=0.1, final_ep=-1):
        self.increment_size = increment_size
        self.discount_factor = discount_factor
        super(incrementLR, self).__init__(optimizer, final_ep)

    def get_lr(self):
        return [base_lr * self.discount_factor ** (self.final_ep // self.increment_size)
                for base_lr in self.first_lr]



class MultiincrementLR(_LRScheduler):

    def __init__(self, optimizer, increase, discount_factor=0.1, final_ep=-1):
      
        self.increase = increase
        self.discount_factor = discount_factor
        super(MultiincrementLR, self).__init__(optimizer, final_ep)

    def get_lr(self):
        return [base_lr * self.discount_factor ** self.increase * self.final_ep for base_lr in self.first_lr]



class ExponentialLR(_LRScheduler):


    def __init__(self, optimizer, discount_factor, final_ep=-1):
        self.discount_factor = discount_factor
        super(ExponentialLR, self).__init__(optimizer, final_ep)

    def get_lr(self):
        return [base_lr * self.discount_factor ** self.final_ep
                for base_lr in self.first_lr]






class StratifiedBootstrap(object):
 

  def __init__(self, *args, random_state = None, bool_bootstrap = False,**kwargs):


    super().__init__(*args, random_state=random_state, **kwargs)
    self.lst_dims = args[0].shape
    self.sequential_tasks = self.lst_dims[1]
    self.task_order = bool_bootstrap
    self.task_index = self.return_sampled_index()

  

  def update_sample(self, num):

    sample_i = np.random.choice(num, self.lst_dims)
    if self.task_order == True:
        task_indices = np.random.choice(self.sequential_tasks, self.task_index[0].shape, replace=True)
        return (sample_i, task_indices, self.task_index[1:])
    
    else:
        return (sample_i, self.task_index)


  def return_sampled_index(self):

    tile_indices = tuple(slice(v) for v in (0, *self.lst_dims[1:]))
    i = np.ogrid[tile_indices]
    return i[1:]


class StratifiedIndependentBootstrap(StratifiedBootstrap):


  def __init__(self, *args, random_state = None, **kwargs: np.ndarray):

    super().__init__(*args, random_state=random_state, **kwargs)
    self.lst_dims = [arg.shape for arg in args]
    self.lst_of_dims_of_args = {key: val.shape for key, val in self._kwargs.items()}
    self.sampled_indices = [self.return_sampled_index(arg_shape) for arg_shape in self.lst_dims]
    self.dict_indices_sample = {key: self.return_sampled_index(kwarg_shape)for key, kwarg_shape in self.lst_of_dims_of_args.items()}


  def return_current_indices(self, num_runs, shape, sampled_ind):

    indices = np.random.choice(num_runs, shape, replace=True)
    return (indices, sampled_ind)

  def return_sampled_index(self, shape):

    tile_indices = tuple(slice(x) for x in (0, *shape[1:]))
    i = np.ogrid[tile_indices]
    return i[1:]

  def update_sample(self):

    pos_indices = [self.return_current_indices(self._num_arg_items, self.lst_dims ,self.sampled_indices)]
    dict_index = {}
    for key in self._kwargs:
      dict_index[key] = self.return_current_indices(self._num_kw_items[key],self.lst_of_dims_of_args[key],
                                          self.dict_indices_sample[key])
    return pos_indices, dict_index


def mean_score_deviation(scores, tau):

  return np.mean(np.mean(scores, axis=0) > tau)

def return_performance_at_increments(metric_list, eval_fun, method, bool_bootstrap=False, repetitions= 50000, conf_int = 0.95, random_state = None):
  flat_est, singular_est = {}, {}
  
  for metric, value in metric_list.items():
    
    if isinstance(value, np.ndarray):
      stratified_bs = StratifiedBootstrap(
         value, bool_bootstrap=bool_bootstrap, random_state=random_state)
      singular_est[metric] = eval_fun(value)
    else:
     
      stratified_bs = StratifiedIndependentBootstrap(value, random_state=random_state)
      
      
      singular_est[metric] = eval_fun(value)
    flat_est[metric] = stratified_bs.conf_int(eval_fun, repetitions=repetitions, size=conf_int, method=method)
  
  return singular_est, flat_est



def run_score_deviation(scores, tau):

  return np.mean(scores > tau)





score_distributions = np.vectorize(run_score_deviation, excluded=[0])
average_score_distributions = np.vectorize(mean_score_deviation, excluded=[0])


def create_performance_profile(
    metric_list,para_list,bool_score,custom_profile_eval_fun, method, bool_bootstrap = False,
    repetitions= 3000, conf_int = 0.95):
  
  if custom_profile_eval_fun is None:

    def profile_eval_funtion(scores):
      if bool_score == True:

        return score_distributions(scores, para_list)

      return average_score_distributions(scores, para_list)
  else:
    profile_eval_funtion = lambda scores: custom_profile_eval_fun(scores, para_list)

  single, multiple = return_performance_at_increments(metric_list,eval_fun=profile_eval_funtion,bool_bootstrap=bool_bootstrap,
      method=method,repetitions=repetitions,conf_int=conf_int)
  
  return single, multiple

#The rest of the util functions are adapted from StableBaselines
def store_args(method):

    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]

        args = defaults.copy()

        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def import_function(spec):

    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def flatten_grads(var_list, grads):
    grads_flattened =  tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)
    return grads_flattened


def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
 
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes)-1 else None
        input = tf.layers.dense(inputs=input,units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse, name=name+'_'+str(i))
        if activation:
            input = activation(input)
            
            
    if flatten:
 
        input = tf.reshape(input, [-1])
    return input


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n, binding="core"):

    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
       
        if platform.system() == 'Darwin':
            args = ["mpirun","-np",
                str(n),sys.executable]
            
            
        else:
            args = ["mpirun","-np",str(n),"-bind-to", binding, "-allow-run-as-root",sys.executable]
        
        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):

    combined_episodes = {}
    for val in episode.keys():
        value = np.array(episode[val]).copy()
        combined_episodes[val] = value.swapaxes(0, 1)

    return combined_episodes


def transitions_in_episode_batch(episode_batch):

    return (episode_batch['u'].shape)[0] * (episode_batch['u'].shape)[1]


def reshape_for_broadcasting(source, target):

    dim = len(target.get_shape())
    shape = ([1] * (dim-1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)

def split_observation_np(env_name, obs):

    if env_name in ['FetchPush-v1', 'FetchSlide-v1', 'FetchPickAndPlace-v1']:
 
        grip_pos, object_pos, _, _, _, _, _, _, _ = np.split(obs, [3, 6, 9, 11, 14, 17, 20, 23], axis=-1)

        obs_achieved_goal = object_pos.copy()
        obs_excludes_goal = grip_pos.copy()

    return (obs_excludes_goal, obs_achieved_goal)

def split_observation_tf(env_name, o):


    if env_name in ['FetchPush-v1', 'FetchSlide-v1', 'FetchPickAndPlace-v1']:

        grip_pos, object_pos, _, _, _, _, _, _, _ = tf.split(o, [3, 3, 3, 2, 3, 3, 3, 3, 2], axis=-1)

     

    return (grip_pos, object_pos)

def make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(ims, filename, lib='cv2'):
    make_dir(filename)
    fps = 30.0
    (height, width, _) = ims[0].shape
    if lib == 'cv2':
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
    elif lib == 'imageio':
        import imageio
        writer = imageio.get_writer(filename, fps=fps)
        
    for i in range(ims.shape[0]):
        if lib == 'cv2':
            writer.write(cv2.cvtColor(np.uint8(ims[i]), cv2.COLOR_RGB2BGR))
        elif lib == 'imageio':
            writer.append_data(ims[i])
    if lib == 'cv2':
        writer.release()
    elif lib == 'imageio':
        writer.close()

def dumpJson(dirname, episodes, epoch, rank):
    os = []
    for episode in episodes:
        episode['o'] = episode['o'].tolist()
        os.append(episode['o'])
    with open(dirname+'/rollout_{0}_{1}.txt'.format(epoch, rank), 'w') as file:
         file.write(json.dumps(os))

def loadJson(dirname, epoch, rank):
    filename = '/rollout_{0}_{1}.txt'.format(epoch, rank)
    with open(dirname+filename, 'r') as file:
        os = json.loads(file.read())
        return os

def save_weight(sess, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    return {v.name: sess.run(v) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ddpg' + '/' + '')}

def load_weight(sess, data, include=[]):
    for s in include:
        for var in tf.global_variables():
            if (var.name in data.keys()) and (s in var.name):
                if var.shape == data[var.name].shape:
                    sess.run(var.assign(data[var.name]))
                    print('load weight: ', var.name)


