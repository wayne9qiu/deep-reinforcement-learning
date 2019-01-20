from dqn_engine import Engine
from dqn_agent import Agent, LiveAgent
from unityagents import UnityEnvironment

def train():
	env = UnityEnvironment(file_name="data/Banana.x86_64")

	brain_name = env.brain_names[0]
	brain = env.brains[brain_name]

	#Number of actions
	action_size = brain.vector_action_space_size

	#Reset the environment
	env_info = env.reset(train_mode=True)[brain_name]

	#Examine the state space 
	state = env_info.vector_observations[0]
	state_size = len(state)

	#Crate a agent
	agent = Agent(state_size=state_size, action_size=action_size, seed=0)

	#Create a game engine
	engine = Engine(env)

	#Start trainging
	scores = engine.train(agent)

def play():
	env = UnityEnvironment(file_name="data/Banana.x86_64")

	brain_name = env.brain_names[0]
	brain = env.brains[brain_name]

	#Number of actions
	action_size = brain.vector_action_space_size

	#Reset the environment
	env_info = env.reset(train_mode=False)[brain_name]

	#Examine the state space 
	state = env_info.vector_observations[0]
	state_size = len(state)

	#Crate a agent
	agent = LiveAgent(state_size, action_size, "checkpoint.pth")

	#Create a game engine
	engine = Engine(env)

	#Start to play
	engine.play(agent)
    
if __name__ == "__main__":
	#train()
	play()
