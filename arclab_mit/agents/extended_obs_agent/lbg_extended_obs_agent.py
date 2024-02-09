# from kspdg.lbg1.lg1_envs import LBG1_LG1_I2_Env
from kspdg.lbg1.lg2_envs import LBG1_LG2_I2_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from arclab_mit.agents.extended_obs_agent.simulate import closest_approach
from arclab_mit.agents.extended_obs_agent.common import single_spacecraft_message, compare_spacecraft_message, round_arr
from arclab_mit.agents.extended_obs_agent.generic_llm_agent import LLMAgent

def obs_to_state(obs):
    return obs[3:6], obs[6:9], obs[9:12], obs[12:15], obs[15:18], obs[18:21]

class LBGExtendedObsAgent(LLMAgent):
    system_prompt_path = "system_prompt_lbg.md"
    def __init__(self):
        super().__init__()
        self.stop_tangential_response = True
        self.start_adjusting_response = True
        self.tangent_acc_time = 25

    def get_manual_response(self, observation):
        state = obs_to_state(observation)
        pursuer_pos, pursuer_vel, lady_pos, lady_vel, guard_pos, guard_vel = state
        pursuer_pos = np.array(pursuer_pos)
        lady_pos = np.array(lady_pos)
        guard_pos = np.array(guard_pos)
        if observation[0] < self.tangent_acc_time:
            guard_relative_pos = pursuer_pos - guard_pos
            tangent_dir = np.array([0.0, 0.0, 1.0]) # hardcoded, true for I2
            guard_relative_pos = round_arr(guard_relative_pos, 2)
            tangent_dir = round_arr(tangent_dir, 2)
            return {
                "role": "assistant",
                "content": 
                " ".join([
                    f"I will accelerate tangent to the relative position to the guard.",
                    f"The relative position is {guard_relative_pos}, and the z-component is roughly 0, so one possible direction is {tangent_dir}.",
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, tangent_dir)) + "]\n}"
                }
            }
        
        if self.stop_tangential_response and observation[0] >= self.tangent_acc_time:
            self.stop_tangential_response = False
            lady_acc_dir = (lady_pos - pursuer_pos)/np.linalg.norm(pursuer_pos - lady_pos)
            final_acc_dir = lady_acc_dir
            final_acc_dir[2] = -0.5
            lady_acc_dir = round_arr(lady_acc_dir, 2)
            final_acc_dir = round_arr(final_acc_dir, 2)
            return {
                "role": "assistant",
                "content": 
                " ".join([
                    f"Now, let's stop accelerating tangetially and start accelerating towards the lady to close the distance.",
                    f"However, we need to also set the z component acceleration to -0.5 to help cancel out the previous tangential acceleration."
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, final_acc_dir)) + "]\n}"
                }
            }
        
        closest_state_lady, closest_time_lady = closest_approach((pursuer_pos, pursuer_vel, lady_pos, lady_vel), 300)
        
        if self.start_adjusting_response and observation[0] >= 65:
            self.start_adjusting_response = False
            pos1, _, pos2, _ = closest_state_lady
            closest_approach_dir = (pos2 - pos1)/np.linalg.norm(pos1 - pos2)
            closest_approach_dir = round_arr(closest_approach_dir, 2)
            return {
                "role": "assistant",
                "content":
                " ".join([
                    f"We are approaching the lady fast enough, so let's adjust our intercept by accelerating based on what the closest approach state tells us.",
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, closest_approach_dir)) + "]\n}"
                }
            }


        return None

    def get_message(self, observation):
        state = obs_to_state(observation)
        pursuer_pos, pursuer_vel, lady_pos, lady_vel, guard_pos, guard_vel = state
        closest_state_lady, closest_time_lady = closest_approach((pursuer_pos, pursuer_vel, lady_pos, lady_vel), 300)
        closest_state_guard, closest_time_guard = closest_approach((pursuer_pos, pursuer_vel, guard_pos, guard_vel), 300)
        
        return "\n".join([
            f"Elapsed Time: {observation[0]} [s]",
            "Current state:",
            single_spacecraft_message(state[0], state[1], "pursuer", use_altitude=False, use_velocity=False),
            single_spacecraft_message(state[2], state[3], "lady", use_altitude=False, use_velocity=False),
            single_spacecraft_message(state[4], state[5], "guard", use_altitude=False, use_velocity=False),
            "Relative to lady:",
            compare_spacecraft_message(state[0:4], "pursuer", "lady", True, use_velocity=False),
            "Relative to guard:",
            compare_spacecraft_message(state[0:2] + state[4:6], "pursuer", "guard", False, use_velocity=False),
            "",

            f"The simulated closest approach between you and the guard will happen in {closest_time_guard}s. Here is some information when that happens:",
            compare_spacecraft_message(closest_state_guard, "pursuer", "guard", False, use_velocity=False),
            "",

            f"The simulated closest approach between you and the lady will happen in {closest_time_lady}s. Here is some information when that happens:",
            compare_spacecraft_message(closest_state_lady, "pursuer", "lady", True, use_velocity=False),
        ])

if __name__ == "__main__":
    my_agent = LBGExtendedObsAgent()    
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=LBG1_LG2_I2_Env, 
        env_kwargs=None,
        runner_timeout=180,
        debug=False)
    runner.run()