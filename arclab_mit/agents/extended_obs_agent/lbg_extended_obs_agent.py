from kspdg.lbg1.lg1_envs import LBG1_LG1_I2_Env
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

    def get_manual_response(self, observation):
        state = obs_to_state(observation)
        pursuer_pos, _, lady_pos, _, guard_pos, _ = state
        pursuer_pos = np.array(pursuer_pos)
        lady_pos = np.array(lady_pos)
        lady_acc_dir = (lady_pos - pursuer_pos)/np.linalg.norm(pursuer_pos - lady_pos)
        guard_escape_dir = (pursuer_pos - guard_pos)/np.linalg.norm(pursuer_pos - lady_pos)
        lady_dir_weight = 0.9
        guard_dir_weight = 0.1
        final_acc_dir = lady_dir_weight*lady_acc_dir + guard_dir_weight*guard_escape_dir
        lady_acc_dir = round_arr(lady_acc_dir, 2)
        guard_escape_dir = round_arr(guard_escape_dir, 2)
        final_acc_dir = round_arr(final_acc_dir, 2)
        return {
            "role": "assistant",
            "content": 
            (
                f"Because of the large distance between us and the lady, we should accelerate towards the lady to close the distance." 
                f"The accelerate direction should be {lady_acc_dir}, so I will accelerate in this geenral direction."
                f"This is also a good choice because even at the simulated closest approach with the lady, this direction is still good, so we are safe to accelerate in this direction."
                f"However, to avoid the guard, I will also accelerate slightly away from the guard."
                f"To do this, I will add a component of {guard_escape_dir} to my acceleration."
                f"My final acceleration will be {lady_dir_weight}*{lady_acc_dir} + {guard_dir_weight}*{guard_escape_dir}, which is {final_acc_dir}."
            ),
            "function_call": {
                "name": "apply_throttle",
                "arguments": "{\n  \"throttle\": [" + ", ".join(map(str, final_acc_dir)) + "]\n}"
            }
        }

    def get_message(self, observation):
        state = obs_to_state(observation)
        pursuer_pos, pursuer_vel, lady_pos, lady_vel, guard_pos, guard_vel = state
        closest_state_lady, closest_time_lady = closest_approach((pursuer_pos, pursuer_vel, lady_pos, lady_vel), 300)
        closest_state_guard, closest_time_guard = closest_approach((pursuer_pos, pursuer_vel, guard_pos, guard_vel), 300)
        
        return "\n".join([
            f"Elapsed Time: {observation[0]} [s]",
            "Current state:",
            single_spacecraft_message(state[0], state[1], "pursuer"),
            single_spacecraft_message(state[2], state[3], "lady"),
            single_spacecraft_message(state[4], state[5], "guard"),
            compare_spacecraft_message(state[0:4], "pursuer", "lady", True),
            compare_spacecraft_message(state[0:2] + state[4:6], "pursuer", "guard", False),
            "",

            f"The simulated closest approach between you and the lady will happen in {closest_time_lady}s). Here is some information when that happens:",
            compare_spacecraft_message(closest_state_lady, "pursuer", "lady", True),
            "",

            f"The simulated closest approach between you and the guard will happen in {closest_time_guard}s). Here is some information when that happens:",
            compare_spacecraft_message(closest_state_guard, "pursuer", "guard", False),
        ])

if __name__ == "__main__":
    my_agent = LBGExtendedObsAgent()    
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=LBG1_LG1_I2_Env, 
        env_kwargs=None,
        runner_timeout=180,
        debug=False)
    runner.run()