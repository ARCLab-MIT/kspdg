import numpy as np
import openai
import os
import json
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env

# Set your OpenAI API key
# openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = "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"


class QLearningAgent(KSPDGBaseAgent):
    def __init__(self, env, state_size, action_size):
        super().__init__()
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            # Explore: choose a random action
            return np.random.choice(self.action_size)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            return np.argmax(self.q_table[state])

    def update_q_table(
        self, state, action, reward, next_state, learning_rate=0.1, discount_factor=0.9
    ):
        # Q-value update using the Q-learning formula
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + learning_rate * (
            reward + discount_factor * max_future_q - current_q
        )
        self.q_table[state, action] = new_q

    def get_action(self, observation):
        # Convert the observation to a state
        state = convert_observation_to_state(observation)

        # Select an action using the Q-learning agent
        action = self.select_action(state)

        # Convert the action to the format expected by the environment
        kspdg_action = {
            "burn_vec": [0.0, 0.0, 0.0, 0.0],  # Adjust this based on your action space
            "vec_type": 0,
            "ref_frame": 0,
        }

        # Update Q-table based on the observed reward
        next_observation, reward, is_done, _ = self.env.step(kspdg_action)
        next_state = convert_observation_to_state(next_observation)
        self.update_q_table(state, action, reward, next_state)

        return kspdg_action


# Function to convert observation into a state representation
def convert_observation_to_state(observation):
    # This is a simple example; you might need to design a more complex state representation
    return hash(tuple(observation))


if __name__ == "__main__":
    # Instantiate the KSPDG environment
    env = PE1_E1_I3_Env()

    # Instantiate the Q-learning agent
    state_size = (
        1  # You may need to adjust the state size based on your observation space
    )
    action_size = 4  # Assuming there are four possible actions
    rl_agent = QLearningAgent(env, state_size, action_size)

    # Train the agent using reinforcement learning
    rl_agent.train(episodes=1000)
