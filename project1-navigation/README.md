[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This repository contains the implementation for the first project of the Udacity deep reinforcement learning nano degree program. The implemented learning algorithm trains an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the trained agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

The project report [report.pdf](https://github.com/wayne9qiu/deep-reinforcement-learning/blob/master/project1-navigation/report.pdf) describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

### Dependencies

To set up the python environment to run the code in this repository, follow the instructions below. The instructions assume Anaconda pytbon 3.6 and up installed in a Mac or Linus platform.

1. Create (and activate) a new environment with Python 3.6. 
	```bash
   	conda create --name drlnd python=3.6
	source activate drlnd
   ```

2. Clone the repository and navigate to the `project1-navigaton/` folder.  Then, install several dependencies.
	```bash
	git clone https://github.com/wayne9qiu/deep-reinforcement-learning.git
	cd deep-reinforcement-learning/project1-navigaton
	pip install ./python
	```

### Instructions

The code only runs on CPU. To run it on GPU, you need to install box2d package, and modify the dqn_agent.py file to import the package.

1. To train the agentm, run the command below.
	```bash
	python dqn_train.py
	```
2. To view the trained agent performing, run the command below 
	```bash
	python dqn_play.py
	```
The image below shows how the trained agent performs.

![Trained Agent][image1]

