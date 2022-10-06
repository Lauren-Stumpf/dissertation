import pickle
import os

from general_util import (save_weight)

import click



def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b

@click.command()
@click.option('--location', type=str, default=None, help='Location of Mutual Information Neural Estimator - saved in logs directory')



def main(location):

    policy_file = location
    
    base = os.path.splitext(policy_file)[0]
    
    with open(policy_file, 'rb') as f:

        pretrain = pickle.load(f)
        
    mutual_information_weights = save_weight(pretrain.sess)
    destination = open(base+'_weight.pkl', 'wb')
    pickle.dump(mutual_information_weights, destination)
    destination.close()

if __name__ == '__main__':
    main()

