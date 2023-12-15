from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from arclab_mit.agents.extended_obs_agent.simulate import closest_approach
from arclab_mit.agents.extended_obs_agent.common import single_spacecraft_message, compare_spacecraft_message, round_arr
from arclab_mit.agents.extended_obs_agent.generic_llm_agent import LLMAgent

def state_to_message(state):
    return "\n".join([
        single_spacecraft_message(state[0], state[1], "pursuer"),
        single_spacecraft_message(state[2], state[3], "evader"),
        compare_spacecraft_message(state, "pursuer", "evader", True),
    ])

def obs_to_state(obs):
    return obs[3:6], obs[6:9], obs[9:12], obs[12:15]

class ExtendedObsAgent(LLMAgent):
    system_prompt_path = "system_prompt.md"

    def get_manual_response(self, observation):
        pursuer_pos, _, evader_pos, _ = obs_to_state(observation)
        pursuer_pos = np.array(pursuer_pos)
        evader_pos = np.array(evader_pos)
        acc_dir = round_arr((evader_pos - pursuer_pos)/np.linalg.norm(pursuer_pos - evader_pos), 1)
        return {
            "role": "assistant",
            "content": f"Because of the large distance between us and the evader, we should accelerate towards the evader to close the distance. The accelerate direction should be {acc_dir}. This is still optimal at the closet approach, so we are safe to accelerate in this direction without overshooting for now.",
            "function_call": {
                "name": "apply_throttle",
                "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, acc_dir)) + "]\n}"
            }
        }

    def get_message(self, observation):
        state = obs_to_state(observation)
        closest_state, closest_time = closest_approach(state, 300)
        return "\n".join([
            f"Elapsed Time: {observation[0]} [s]",
            "Current state:",
            state_to_message(state),
            "",
            f"If your spacecraft and the evader both stop accelerating, this will be the simulated closest approach (happens at time {closest_time}s):",
            state_to_message(closest_state),
        ])

if __name__ == "__main__":
    my_agent = ExtendedObsAgent()    
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E3_I3_Env, 
        env_kwargs=None,
        runner_timeout=120,
        debug=False)
    runner.run()