from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from arclab_mit.agents.extended_obs_agent.simulate import closest_approach
from arclab_mit.agents.extended_obs_agent.common import single_spacecraft_message, compare_spacecraft_message, round_arr
from arclab_mit.agents.extended_obs_agent.generic_llm_agent import LLMAgent

def state_to_message(state, extra_label=""):
    return "".join([
        single_spacecraft_message(state[0], state[1], "pursuer", use_altitude=False, use_velocity=False),
        single_spacecraft_message(state[2], state[3], "evader", use_altitude=False, use_velocity=False),
        compare_spacecraft_message(state, "pursuer", "evader", extra_label=extra_label),
    ])

def obs_to_state(obs):
    return obs[3:6], obs[6:9], obs[9:12], obs[12:15]

class ExtendedObsAgent(LLMAgent):
    system_prompt_path = "system_prompt.md"

    def __init__(self):
        super().__init__()
        self.first_response = True
        self.near_evader_response = True
        self.slow_down_response = False

    def get_manual_response(self, observation):
        state = obs_to_state(observation)
        pursuer_pos, pursuer_vel, evader_pos, evader_vel = state
        if self.first_response:
            self.first_response = False
            pursuer_pos = np.array(pursuer_pos)
            evader_pos = np.array(evader_pos)
            acc_dir = (evader_pos - pursuer_pos)/np.linalg.norm(pursuer_pos - evader_pos)
            final_acc_dir = 0.7 * acc_dir
            acc_dir = round_arr(acc_dir, 2)
            final_acc_dir = round_arr(final_acc_dir, 2)
            return {
                "role": "assistant",
                "content":
                " ".join([
                    f"Because of the large distance between us and the evader, we should accelerate towards the evader to close the distance.",
                    f"Let's accelerate at 70% of max throttle. Our acceleration should be 0.7 * {acc_dir} = {final_acc_dir}.",
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, acc_dir)) + "]\n}"
                }
            }
        closest_state, closest_time = closest_approach((pursuer_pos, pursuer_vel, evader_pos, evader_vel), 300)
        if self.near_evader_response and 10 <= closest_time <= 60:
            self.near_evader_response = False
            pos1, _, pos2, _ = closest_state
            closest_approach_dir = (pos2 - pos1)/np.linalg.norm(pos1 - pos2)
            final_acc_dir = 1.0 * closest_approach_dir
            closest_approach_dir = round_arr(closest_approach_dir, 2)
            final_acc_dir = round_arr(final_acc_dir, 2)
            return {
                "role": "assistant",
                "content":
                " ".join([
                    f"We are getting close to the evader.",
                    f"We need to stop accelerating towards the current evader position.",
                    f"Instead, let's intercept the evader by accelerating in the direction of what the closest approach state tells us, which is {final_acc_dir}.",
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, final_acc_dir)) + "]\n}"
                }
            }
        if self.slow_down_response and 10 <= closest_time <= 30:
            self.slow_down_response = False
            pos1, vel1, pos2, vel2 = closest_state
            closest_approach_dir = (pos2 - pos1)/np.linalg.norm(pos1 - pos2)
            relative_vel_dir = (vel1 - vel2)/np.linalg.norm(vel1 - vel2)
            final_acc_dir = 0.4 * closest_approach_dir - 0.6 * relative_vel_dir
            closest_approach_dir = round_arr(closest_approach_dir, 2)
            relative_vel_dir = round_arr(relative_vel_dir, 2)
            final_acc_dir = round_arr(final_acc_dir, 2)
            return {
                "role": "assistant",
                "content":
                " ".join([
                    f"We should continue to accelerate based on what the cloeset approach state tells us, which is {closest_approach_dir}.",
                    f"However, since we are very close to the evader, we also need to reduce the relative velocity, so we need to accelerate opposite of the relative velocity direction.",
                    f"Let's accelerate by 0.4 * {closest_approach_dir} - 0.6 * {relative_vel_dir} = {final_acc_dir}.",
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, final_acc_dir)) + "]\n}"
                }
            }
        return None

    def get_message(self, observation):
        state = obs_to_state(observation)
        closest_state, closest_time = closest_approach(state, 300)
        return "\n".join([
            f"Elapsed Time: {observation[0]} [s]",
            "Current state:",
            state_to_message(state),
            "",
            f"If your spacecraft and the evader both stop accelerating, this will be the simulated closest approach (happens at time {closest_time}s):",
            state_to_message(closest_state, extra_label=" at closest approach"),
        ])

if __name__ == "__main__":
    my_agent = ExtendedObsAgent()    
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E3_I3_Env, 
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()

"""
292, 284, 285, 19, 29
804, 181, 125, 44, 38, 32
"""