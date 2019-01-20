import numpy as np
from collections import deque

import torch

class Engine():
    """A container to run training episodes"""

    def __init__(self, env):
        """Initialize an container object
        
        Params
        ======
            env: unity environment object
        """

        # initialzie state_size and action_size property of the environment
        self.env = env
        self.brain_name = env.brain_names[0]
    
    def train(self, agent, episodes=2000, max_steps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, model_path='model.pth'):
        """ Train the Deep Q-Learning model
    
        Params
        ======
            agent: the DDQN agent 
            episodes (int): maximum number of training episodes
            max_step (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
			model_path: the path to save the trained model
        """

        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon

        for i_episode in range(1, episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0

            for t in range(max_steps):
                action = agent.act(state, eps)

                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]  

                agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward

                if done:
                    break 

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score

            eps = max(eps_end, eps_decay*eps) # decrease epsilon

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if np.mean(scores_window) >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                agent.save_model(model_path=model_path)
                break

        self.env.close()

        return scores

    def play(self, agent):
        """Use trained agent to play
    
        Params
        ======
            agent: the trained DDQN agent
        """
		
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        state = env_info.vector_observations[0]
        score = 0

        while True:
            action = agent.act(state)
            env_info = self.env.step(action)[self.brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]

            state = next_state
            score += reward

            if done:
                break
        
        self.env.close()

        print("Score: {}".format(score)) 
