from dqn_engine import Engine
from dqn_agent import LiveAgent
from unityagents import UnityEnvironment

SAVED_MODEL = "model.pth"
ENV_FILE = "data/Banana.x86_64"

def play():
	# Initialize the Unity Environment
	env = UnityEnvironment(file_name=ENV_FILE)

	brain_name = env.brain_names[0]
	brain = env.brains[brain_name]

	# Number of actions
	action_size = brain.vector_action_space_size

	# Reset the environment
	env_info = env.reset(train_mode=False)[brain_name]

	# Examine the state space 
	state = env_info.vector_observations[0]
	state_size = len(state)

	# Crate a agent
	agent = LiveAgent(state_size, action_size, SAVED_MODEL)

	# Create a game engine
	engine = Engine(env)

	# Start to play
	engine.play(agent)
    
if __name__ == "__main__":
	play()
