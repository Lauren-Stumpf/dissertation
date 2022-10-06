
### MAIN 



import json 
import numpy as np

print('Hello. This is the codebase for Using Intrinsically Motivated Reinforcement Learning Agents')



environment = None
while environment == None: 
    print('Please enter the number of the environment you would like to train - either 1, 2 or 3')
    print("1. Fetch Push")
    print("2. Fetch PickAndPlace")
    print("3. Fetch Slide")
    
    x = input()
    try:
        if str(1) in x: 
            environment = 'FetchPush-v1'
            print('The environment is Fetch Push')
        elif str(2) in x: 
            environment = 'FetchPickAndPlace-v1'
            print('The environment is Fetch PickAndPlace')
            
        elif str(3) in x:
            environment = 'FetchSlide-v1'
            print('The environment is Fetch Slide')
            
            
        else:
            print('Sorry, that was not recognised')
    
    except:
        if  x == 1: 
            environment = 'FetchPush-v1'
            print('The environment is Fetch Push')
        elif x == 2: 
            environment = 'FetchPickAndPlace-v1'
            print('The environment is Fetch PickAndPlace')
            
        elif x == 3:
            environment = 'FetchSlide-v1'
            print('The environment is Fetch Slide')
            
        else:
            print('Sorry, that was not recognised')
        
    
    

recognised = False

while recognised == False:
    print('Please enter the number of CPU cores avaliable to train the agent, we recommend 32. A lower number will affect performance')
    try:
        number_CPUs = int(input())
        recognised = True
    except:
        print('Sorry, that was not recognised')
        
print('The number of CPUs to be used is ' + str(number_CPUs))
print('We have created two parameters files. To train the Mutual Information Neural Estimator please run train.py --parameters MINE_parameters')
print('Then run save_weight.py --location file_location_of_trained_MINE (which can be found in the logs folder with the time and date that train.py was run)')
print('Then run train.py --parameters agent_parameters --location file_location_of_trained_MINE')

#'MINE_parameters
#agent_parameters

environment = environment 
number_CPUs = int(np.floor(number_CPUs / 2))
#cmd line params 
default = {'env': environment,
'num_cpu': number_CPUs,
'n_epochs': 50,
'seed': 0,
'policy_save_interval': 1,
'n_cycles': 50,
'her_buffer_strategy': 'future',
'clip_return': 0,
'binding': 'core',
'logging': True, 
'version': 0,
'note': None }

#default params
DEFAULT_PARAMS = {
    'reward_scaling': 0,
    'weight_scheduler': [(0,0.2),(24,0.2),(200,0.2)],
    'note': 'SAC+MUSIC',
    'normalizer_small_constant': 0.01,
    'load_weight': None,
    'layers': 3,
    'mutual_information_normalization_coefficient': 5000,
    'network_width': 256,
    'rollout_batch_size': 2,
    'noise_constant': 0.2,
    'polyak': 0.95,
    'number_of_goals_to_generate_in_buffer': 'Zero',
    'maximum_action_value': 1.,
    'network_policy_value':'agent:Policy_Value_Function_Structure',
    'input_dimensions': 4,
    'small_constant_for_exploration': 0.3,
    'stochastic': True,
    'normalization_clip_value': 5,
    'policy_learning_rate': 0.001,
    'refine_policy': False,
    'negative_train_of_mutual_information': False,
    'her_buffer_strategy': 'future',
    'schedule_for_mutual_information': [(0,1),(50,1),(200,1)],
    'clip_observations': 200.,
    'Q_function_learning_rate': 0.001,
    'maximum_number_of_epochs': 200,
    'mutual_information_learning_rate': 0.001,
    'n_test_rollouts': 10,
    'skill_learning_rate': 0.001,
    'mutual_information_use_in_buffer': False,
    'regularisation_coefficient': 1.0,
    'n_cycles': 50,
    'scope': 'ddpg',
    'collect_data': False,
    'learn_multiple_skills': 0,
    'collect_video': False,
    'reward_scale': 0.02,
    'test_with_polyak': False,
    'buffer_size': int(1E6),
    'refine_multiple_skills': None,
    'batch_size': 256,
    'network_class_ir':'intrinsic_rewards_estimator:Estimator_Of_Intrinsic_Rewards',
    'num_skills': 1,
    'n_batches': 40,
    'relative_goals': False,
    'her_replay_parameter': 0
}

#Mutual Information Neural Estimator params
MINE_params = {
'env_name': environment,
'num_cpu': number_CPUs,
'n_epochs': 50,
'seed': 0,
'policy_save_interval': 1,
'n_cycles': 50,
'her_buffer_strategy': 'future',
'clip_return': 0,
'binding': 'core',
'logging': True, 
'version': 0,
'note': None,
"stochastic": 'true', 
"her_replay_parameter": 0, 
"reward_scaling": 0, 
"mutual_information_normalization_coefficient": 5000, 
"learn_multiple_skills": 0, 
"reward_scale": 0.02, 
"schedule_for_mutual_information": [[0,1],[50,1],[200,1]], 
"maximum_number_of_epochs":200, 
"number_of_goals_to_generate_in_buffer": "Zero", 
"num_skills": 1, 
"weight_scheduler": [[0,0.2],[24,0.2],[200,0.2]], 
"refine_policy": False, 
"negative_train_of_mutual_information": False, 
"mutual_information_use_in_buffer": False, 
"load_weight": False}

#Training the agent

agent_params = {
'env_name': environment,
'num_cpu': number_CPUs,
'n_epochs': 50,
'seed': 0,
'policy_save_interval': 1,
'n_cycles': 50,
'her_buffer_strategy': 'future',
'clip_return': 0,
'binding': 'core',
'logging': True, 
'version': 0,
'note': None, 
"stochastic": 'true', 
"her_replay_parameter": 0, 
"reward_scaling": 1, # 
"mutual_information_normalization_coefficient": 5000, 
"learn_multiple_skills": 0, 
"reward_scale": 0.02, 
"schedule_for_mutual_information": [[0,1],[50,1],[200,1]], 
"maximum_number_of_epochs":200, 
"number_of_goals_to_generate_in_buffer": "Env", # 
"num_skills": 1, 
"weight_scheduler": [[0,0.2],[24,0.2],[200,0.2]], 
"refine_policy": False, 
"negative_train_of_mutual_information": True, # 
"mutual_information_use_in_buffer": False, 
"load_weight": None} #     



for (key, value) in DEFAULT_PARAMS.items():
    if key in MINE_params:
        pass
    else:
        MINE_params[key] = value
 


for (key, value) in DEFAULT_PARAMS.items():
    if key in agent_params:
        pass
    else:
        agent_params[key] = value
        

with open('MINE_parameters.json', 'w') as f:
        json.dump(MINE_params, f)
        
with open('agent_parameters.json', 'w') as f:
        json.dump(agent_params, f)

