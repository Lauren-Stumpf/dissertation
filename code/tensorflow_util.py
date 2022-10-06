
####  TENSORFLOW UTIL

import numpy as np
import tensorflow as tf  # pylint: ignore-module
import os
import multiprocessing

#Taken from StableBaselines
def var_shape(x):
    out = x.get_shape().as_list()

    return out
#Taken from StableBaselines
def numel(x):
    return intprod(var_shape(x))

#Taken from StableBaselines
def intprod(x):
    return int(np.prod(x))

#Taken from StableBaselines
def make_session(num_cpu=None, make_default=False, graph=None):

    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
        
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    
    tf_config.gpu_options.allocator_type = 'BFC'
    
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    return tf.Session(config=tf_config, graph=graph)

#Taken from StableBaselines
def single_threaded_session():
    return make_session(num_cpu=1)




def shape(x):
  if isinstance(x, tf.SparseTensor):
    return x.dense_shape
  else:
    return tf.shape(input=x, out_type=tf.int64)


def reshape(x, shape):

  return (tf.sparse.reshape(x, shape) if isinstance(x, tf.SparseTensor)
          else tf.reshape(x, shape))


def squeeze(x, axis):

  if isinstance(x, tf.SparseTensor):
    
    return tf.SparseTensor(indices=tf.concat((x.indices[:, :axis], x.indices[:, axis + 1:]),
            axis=1),
        values=x.values,dense_shape=tf.concat((x.dense_shape[:axis], x.dense_shape[axis + 1:]),
            axis=0))
  else:
    return tf.squeeze(x, [axis])


def expand_dims(x, axis):

  if isinstance(x, tf.SparseTensor):
    
    not_empty_num = (x.indices.shape[0] or tf.get_static_shape(x.dense_shape)[0] or tf.shape(x.indices)[0])
    dim = tf.cast(x.shape, tf.int64)
    return tf.SparseTensor(
        indices=tf.concat((tf.zeros([not_empty_num, 1], dtype=tf.int64),x.indices),
                          axis=1),
        values=x.values,
        dense_shape=tf.concat(([1], dim), axis=0))
  else:
    return tf.expand_dims(x, axis)


def slice_from(x, axis, start):

  if start != None:
    if not tf.is_tensor(start) and start < 0:
      start = x.dense_shape[axis] + start
    first = tf.reshape(tf.where(x.indices[:, axis] >= start),
        [-1])
    non_zero_indices = tf.gather(x.non_zero_indices, first)
    
    non_zero_indices = tf.unstack(non_zero_indices, axis=1)
    non_zero_indices = tf.stack(non_zero_indices[:axis]
                       + [non_zero_indices[axis] - start]
                       + non_zero_indices[axis + 1:],
                       axis=1)
    transformed_dim = tf.unstack(x.dense_shape)
    
    
    transformed_dim[axis] = transformed_dim[axis] - start
    return tf.SparseTensor( non_zero_indices=non_zero_indices, values=tf.gather(x.values, first), dense_shape=tf.stack(transformed_dim))
  else:
    ndims = len(x.shape)
 
    slices = tuple([slice(None)] * axis
                   + [slice(start, None)]
                   + [slice(None)] * (ndims - axis - 1))
    return x[slices]


def slice_to(x, axis, end):

  if end != None:
    if not tf.is_tensor(end) and end < 0:
      end = x.dense_shape[axis] + end
    first = tf.reshape(tf.where(x.indices[:, axis] < end), [-1])
    
    transformed_dim = tf.unstack(x.dense_shape)
    transformed_dim[axis] = end
    
    
    return tf.SparseTensor(non_zero_indices=tf.gather(x.indices, first),
        values=tf.gather(x.values, first),
        dense_shape=tf.stack(transformed_dim))
  else:
    ndims = len(x.shape)

    slices = tuple([slice(None)] * axis + [slice(None, end)]
                   + [slice(None)] * (ndims - axis - 1))
    return x[slices]



