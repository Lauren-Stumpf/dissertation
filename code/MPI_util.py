


## MPI UTIL

import time 
from mpi4py import MPI
import numpy as np
import tensorflow as tf



#Adapted from Mutual Information State Intrinsic Control 
def mpi_mean(v, axis=0, comm=None, keepdims=False):
    v = np.asarray(v)

    if comm is None: 
        comm = MPI.COMM_WORLD
        
    total_sum = v.sum(axis=axis, keepdims=keepdims)
    dim = total_sum.size
    
    part_sum = np.zeros(dim+1, v.dtype)
    
    part_sum[:dim] = total_sum.ravel()
    
    part_sum[dim] = v.shape[axis]
    
    end_sum = np.zeros_like(part_sum)
    
    comm.Allreduce(part_sum, end_sum, op=MPI.SUM)
    return end_sum[:dim].reshape(total_sum.shape) / end_sum[dim], end_sum[dim]

#Adapted from Mutual Information State Intrinsic Control 
def mpi_moments(v, axis=0, comm=None, keepdims=False):
    v = np.asarray(v)

    
    avg, number = mpi_mean(v, axis=axis, comm=comm, keepdims=True)
    squared_std, std_count = mpi_mean(np.square(v - avg), axis=axis, comm=comm, keepdims=True)
    
    
    std = np.sqrt(squared_std)
    
    if keepdims:
        return avg, std, number
    
    else:
        dims = avg.shape[:axis] + avg.shape[axis+1:]
        avg = avg.reshape(dims)
        std = std.reshape(dims)
        return avg, std, number
    
    


def pprint(string, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        print(string)

def distributed_operation(size):
    comm = MPI.COMM_WORLD

    mpi_rows = int(np.floor(np.sqrt(size)))
    mpi_cols = size // mpi_rows
    

    ccomm = comm.Create_cart( (mpi_rows, mpi_cols), periods=(True, True), reorder=True)

    pi_row, pi_col = ccomm.Get_coords( ccomm.rank ) 
    lst_res = [0,0,0,0]
    
    lst_res[0], lst_res[1] = ccomm.Shift(0, 1)
    lst_res[2],  lst_res[3]  = ccomm.Shift(1, 1)

    container_1 = np.random.normal(size=(3000, 3000)).astype(np.float32)
    container_2 = np.random.normal(size=(3000, 3000)).astype(np.float32)
    container_3 = np.zeros_like(container_1)

    object_1 = container_1
    object_2 = container_2
    
    
    object_1_ = np.empty_like(container_1)
    object_2_ = np.empty_like(container_1)
    req = [None, None, None, None]

    t0 = time()
    for r in range(mpi_rows):
        req[0]  = ccomm.Isend(object_1 , lst_res[0])
        req[1]  = ccomm.Irecv(object_1_, lst_res[1])
        req[2] = ccomm.Isend(object_2 , lst_res[2])
        req[3] = ccomm.Irecv(object_2_, lst_res[3])

        container_3 += np.dot(object_1, object_2)

        req[0].Waitall(req)
   
    comm.barrier()
    t_total = time()-t0

    t0 = time()
    np.dot(object_1, object_2)
    t_serial = time()-t0


    pprint("Computed (serial) %d x %d x %d in  %6.2f seconds" % (3000, 3000, 3000, t_serial))
    pprint(" ... expecting parallel computation to take %6.2f seconds" % (mpi_rows*mpi_rows*mpi_cols*t_serial / comm.size))
    pprint("Computed (parallel) %d x %d x %d in        %6.2f seconds" % (mpi_rows*3000, mpi_rows*3000, mpi_cols*3000, t_total))

    comm.barrier()
    
    




def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)



def distributed_task_pull(tag, num_workers, tasks):
    
    tags = enum(tag)

    comm = MPI.COMM_WORLD         
    closed_workers = 0
    status = MPI.Status()   
    
    task_index = 0

    
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if tag == tags.START:
            result = task**2
            comm.send(result, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break
        
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
           
            if task_index < len(tasks):
                comm.send(tasks[task_index], dest=source, tag=tags.START)
                print("Sending task %d to worker %d" % (task_index, source))
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            results = data
         
        elif tag == tags.EXIT:

            closed_workers += 1

    comm.send(None, dest=0, tag=tags.EXIT)
    
    return results



def numel(x):
    return int(np.prod(var_shape(x)))



def var_shape(x):
    out = x.get_shape().as_list()
 
    return out



class RestructuredObject(object):
    def __init__(self, objects):
        assigns = []
        dims = list(map(var_shape, objects))
        sum_total = np.sum([int(np.prod(dim)) for dim in dims])

        self.variable_object = variable_object = tf.placeholder(tf.float32, [sum_total])
        count = 0
        
        for (dim, v) in zip(dims, objects):
            s = int(np.prod(dim))
            assigns.append(tf.assign(v, tf.reshape(variable_object[count:count + s], dim)))
            count += s
            
        self.op = tf.group(*assigns)

    def __call__(self, variable_object):
        tf.get_default_session().run(self.op, feed_dict={self.variable_object: variable_object})
        
        

#Adapted from Mutual Information State Intrinsic Control 
class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.restructuredObject = RestructuredObject(var_list)
        self.returnObject = ReturnObject(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm
        
    def check_synced(self):
        if self.comm.Get_rank() == 0: 
            theta = self.returnObject()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.returnObject()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.restructuredObject(self.returnObject() + step)

    def sync(self):
        theta = self.returnObject()
        self.comm.Bcast(theta, root=0)
        self.restructuredObject(theta)

    
            
            
class ReturnObject(object):
    def __init__(self, objects):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in objects])

    def __call__(self):
        return tf.get_default_session().run(self.op)



