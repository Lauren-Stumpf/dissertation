# Deep Ensembled Truncated Quantile Critic with Recurrent Networks (DETRN+) - Reinforcement Learning:

This work was undertaken and submitted as a 3rd year Bachelors Dissertation/Project as part of the degree of Bsc Natural Sciences to the Board of Examiners in the Department of Computer Sciences, Durham University, and is licensed accordingly. 
## Grade: 1st - 84/100, 11th best in year (of 103 projects).

This agent was trained and extensively tested on the Fetch Environments, credits for the enviroment go to [Matthias Plappert](https://github.com/matthiasplappert). See main paper and codebase for all references and accreditatons.

## Demo video :
  > <img src="https://github.com/Lauren-Stumpf/dissertation/blob/main/videos/fetch_pick_and_place_video.gif" width="250" height="250"/>
  > <img src="https://github.com/Lauren-Stumpf/dissertation/blob/main/videos/fetch_push_video.gif" width="250" height="250"/>
  > <img src="https://github.com/Lauren-Stumpf/dissertation/blob/main/videos/fetch_reach_video.gif" width="250" height="250"/>
  > <img src="https://github.com/Lauren-Stumpf/dissertation/blob/main/videos/fetch_slide_video.gif" width="250" height="250"/>
  > 
  > During my Dissertation, I surveyed the field of Reinforcement Learning and spent my project investigating how best to combine many recent advances from Reinforcement Learning. In doing so, I created an AI agent capable of teaching itself to complete the Fetch environments (various robotic manipulation tasks) despite the sparse and uninformative reward signal. 
  > 




## Paper Abstract:
### Context:
Manipulation robots are an integral part of manufacturing. However, the current generation of manipulation robots,
struggle to adapt to changes in their environment because they have been explicitly programmed. If instead we used a reinforcement
learning approach, the robot would be more capable of operating in less structured environments and would indeed be able to adapt to
changes in their environment. However it is very difficult to engineer a well-shaped and dense reward signal to allow the agent to learn
a manipulation task. Therefore, the rewards that the robot learns from are usually infrequent (sparse) and uninformative.
### Aims:
To produce a reinforcement learning agent capable of completing various manipulation tasks in spite of the sparse rewards.
Specifically, we hope to produce a skillful agent, through combining and researching state of the art practices and architecture. This
agent should be able to generalise across environments.
### Method:
We supplement the sparse task reward with a dense task-independent intrinsic reward. We then research and combine the
most appropriate recent state of the art namely ensemble learning, hindsight experience replay, D2RL, R2D2 with some novel
practices and architecture such as a new data augmentation scheme and dropout layers.
### Results:
 We produce a Deep Ensembled Truncated Quantile Critic with Recurrent Networks which we define in this paper. The final
agent is extremely competent and surpasses the performance of all surveyed approaches. It is able to complete the three manipulation
tasks to varying degrees of success. It is also able to generalise across environments. We also demonstrate some of the best practices
to exploit a predominately intrinsic reward signal.
### Conclusions:
We found that all included components bolster the performance of the agent with the exception of recurrent networks
which are harmful to the generalisation power of the agent. The novel use of dropout and its impact on generalisation power is
particularly promising. We also gain some valuable insights into how best exploit intrinsic rewards, namely that intrinsic rewards are
best maximised by behaving less greedily and that stochastic agents work best. Overall, the generalisation power of our agent shows
the value that reinforcement learning has in producing a flexible agent that does not put a precise requirements on the environment it is
operating in. 





## Repository Contents:
* The main 20 page paper, Using Intrinsically Motivated Reinforcement Learning for Manipulation Robotics.;
* The Project Demonstration (a 30 minute Oral Exam demonstrating the work undertaken and success of the Project);
* The accompanying codebase for the agent DETRN+, with switches to turn on and off every individual component (very useful for learning what individual papers have contributed, if you are new to the field);
* Videos of the final agent on the Fetch Environments (Fetch-Reach, Fetch-Push, Fetch-Slide, Fetch-Pick-And-Place).

## Neural Network Architecture/Pipeline:
> ![image](https://github.com/Lauren-Stumpf/dissertation/blob/main/photos/neural_net_arch.png)
