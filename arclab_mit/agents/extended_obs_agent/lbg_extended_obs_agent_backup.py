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
        self.first_response = True
        self.second_response = True
        self.passed_guard_response = True
        self.near_evader_response = True

    def get_manual_response(self, observation):
        state = obs_to_state(observation)
        pursuer_pos, pursuer_vel, lady_pos, lady_vel, guard_pos, guard_vel = state
        if self.first_response or self.second_response:
            if self.first_response: self.first_response = False
            else: self.second_response = False
            pursuer_pos = np.array(pursuer_pos)
            lady_pos = np.array(lady_pos)
            guard_pos = np.array(guard_pos)

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

            """
            lady_acc_dir = (lady_pos - pursuer_pos)/np.linalg.norm(pursuer_pos - lady_pos)
            guard_escape_dir = (pursuer_pos - guard_pos)/np.linalg.norm(pursuer_pos - guard_pos)
            lady_dir_weight = 0.8
            guard_dir_weight = 0.2
            final_acc_dir = lady_dir_weight*lady_acc_dir + guard_dir_weight*guard_escape_dir
            lady_acc_dir = round_arr(lady_acc_dir, 2)
            guard_escape_dir = round_arr(guard_escape_dir, 2)
            final_acc_dir = round_arr(final_acc_dir, 2)
            return {
                "role": "assistant",
                "content": 
                " ".join([
                    f"Because of the large distance between us and the lady, we should accelerate towards the lady to close the distance." ,
                    f"However, I will also accelerate away from the guard.",
                    f"My final acceleration will be {lady_dir_weight}*{lady_acc_dir} + {guard_dir_weight}*{guard_escape_dir}, which is {final_acc_dir}.",
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, final_acc_dir)) + "]\n}"
                }
            }
            """
        
        closest_state_lady, closest_time_lady = closest_approach((pursuer_pos, pursuer_vel, lady_pos, lady_vel), 300)
        closest_state_guard, closest_time_guard = closest_approach((pursuer_pos, pursuer_vel, guard_pos, guard_vel), 300)
        if self.passed_guard_response and observation[0] >= 20 and closest_time_guard <= 1.0:
            self.passed_guard_response = False
            pos1, _, pos2, _ = closest_state_lady
            closest_approach_dir = (pos2 - pos1)/np.linalg.norm(pos1 - pos2)
            final_acc_dir = 0.5 * closest_approach_dir
            closest_approach_dir = round_arr(closest_approach_dir, 2)
            final_acc_dir = round_arr(final_acc_dir, 2)
            return {
                "role": "assistant",
                "content":
                " ".join([
                    f"We have just passed the guard's closest position, so we don't need to worry about the guard anymore.",
                    f"We should intercept the lady by accelerating based on what the cloeset approach state tells us, which is {closest_approach_dir}.",
                    f"To not over-adjust, I will set my final acceleration to be 0.6*{closest_approach_dir} = {final_acc_dir}.",
                ]),
                "function_call": {
                    "name": "apply_throttle",
                    "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, final_acc_dir)) + "]\n}"
                }
            }
        
        # [...]
        # closest_time_lady bigger than 10 to avoid triggering at the start (when its 0)
        if self.near_evader_response and 10 <= closest_time_lady <= 40 and observation[0] >= 40:
            self.near_evader_response = False
            pos1, _, pos2, _ = closest_state_lady
            closest_approach_dir = round_arr((pos2 - pos1)/np.linalg.norm(pos1 - pos2), 2)
            return {
                "role": "assistant",
                "content":
                " ".join([
                    f"We are approaching the lady." ,
                    f"We should intercept the lady by accelerating based on what the cloeset approach state tells us, which is {closest_approach_dir}.",
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
            single_spacecraft_message(state[0], state[1], "pursuer", use_velocity=False),
            single_spacecraft_message(state[2], state[3], "lady", use_velocity=False),
            single_spacecraft_message(state[4], state[5], "guard", use_velocity=False),
            compare_spacecraft_message(state[0:4], "pursuer", "lady", True, use_velocity=False),
            compare_spacecraft_message(state[0:2] + state[4:6], "pursuer", "guard", False, use_velocity=False),
            "",

            f"The simulated closest approach between you and the guard will happen in {closest_time_guard}s). Here is some information when that happens:",
            compare_spacecraft_message(closest_state_guard, "pursuer", "guard", False, use_velocity=False),
            "",

            f"The simulated closest approach between you and the lady will happen in {closest_time_lady}s). Here is some information when that happens:",
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