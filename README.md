[//]: # (Image References)

[image1]: https://github.com/aldebaransearch/udacity_drl_project_3/blob/main/tennis.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

In this project, we train a deep reinforcement learning agent to play in the Unity environment Tennis. The report describing the solution in greater detail can be found in [report_3.pdf](https://github.com/aldebaransearch/udacity_drl_project_3/blob/main/report_3.pdf)

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

### State and Action Space
The state space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Solution
The task is episodic. The agents are considered successful in the current setting, when the max score reaches an average score of 0.5 over 100 episodes. However, instead of stopping the agent at a score of 0.5, we run them a bit further to judge different training characteristics.

### Getting Started
**`1`** Build a conda environment using the environment.yml file in this repository by running "conda env create -f environment.yml"

**`2`** Download the environment from from the link below that matches your system:
   **_Linux_**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   **_Mac OSX_**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   **_Windows (32-bit)_**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   **_Windows (64-bit)_**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    

Place the file in the project folder, and unzip it. 

**`3`** Download the notebooks Continuous_Control_1.ipynb and Continuous_Control_20.ipynb as well as the python file util.py.

### Training
Simply follow the instructions in the first half of the Tennis.ipynb notebook. Rewards are saved in a file in a folder `results` (if that does not exist, you should create it) and checkpoints with neural network weights are saved at every 10th episode in the folder `checkpoints` (if that folder does not exist, you should create it).

### Check Solutions
To assess results from training, follow the instructions in the last half of the Tennis.ipynb notebook. 

If you want to examine precalculated solutions, download the `results` and `checkpoints` folders including all their contents and use that instead of your own training results.

### Important note for Linux users
Due to what seems to be a bug in the Unity environment running on Linux machines, the notebook kernel has to be restarted between different trainings or different model assessments.