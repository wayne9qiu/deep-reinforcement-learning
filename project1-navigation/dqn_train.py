from dqn_engine import Engine
from dqn_agent import Agent, LiveAgent
from unityagents import UnityEnvironment

EPISODES = 2000
MAX_STEPS = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
MODEL_PATH = "model.pth"

ENV_FILE = "data/Banana.x86_64"

def train():
	# Initialize Unity Environment
	env = UnityEnvironment(file_name=ENV_FILE)

	brain_name = env.brain_names[0]
	brain = env.brains[brain_name]

	# Number of actions
	action_size = brain.vector_action_space_size

	# Reset the environment
	env_info = env.reset(train_mode=True)[brain_name]

	# Examine the state space 
	state = env_info.vector_observations[0]
	state_size = len(state)

	# Crate a agent
	agent = Agent(state_size=state_size, action_size=action_size, seed=0)

	# Create a game engine
	engine = Engine(env)

	# Start trainging
	scores = engine.train(agent, episodes=EPISODES, max_steps=MAX_STEPS, eps_start=EPS_START, \
	                             eps_end=EPS_END, eps_decay=EPS_DECAY, model_path=MODEL_PATH)

if __name__ == "__main__":
	train()
