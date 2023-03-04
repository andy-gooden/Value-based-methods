[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, we trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started
#### Setting Up Your Python Environment
This code was developed using Python 3.6.15

To manage the Python environment, I used [pyenv](https://github.com/pyenv/pyenv#installation) coupled with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv#installation) 
1. Install Python 3.6.15
    - `pyenv install 3.6.15`
2. Create a virtual environment called `p1_navigation` that uses Python 3.6.15:
    - `pyenv virtualenv 3.6.15 p1_navigation`
    - Note: you can create the environment with a different name, just replace `p1_navigation` in the previous command AND in the `.python-version` file in the root directory
3. Install the required packages:
    - `pip install -r requirements.txt`

#### Setting Up The Unity Environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions
The three main files are:
1. `dqn_agent.py` - this file has the bulk of the hyperparameters to set, and contains code for the agent
2. `model.py` - this file implements the neural network used in the DQN
3.  `Navigation.ipynb` - the Jupyter Notebook that runs it all

Run `Navigation.ipynb` to train your own agent!  
