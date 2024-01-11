from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from arclab_mit.agents.extended_obs_agent.simulate import closest_approach

def obs_to_state(obs):
    return obs[3:6], obs[6:9], obs[9:12], obs[12:15]

class DeterministicAgent(KSPDGBaseAgent):
    def get_action(self, observation):
        state = obs_to_state(observation)
        pursuer_pos, pursuer_vel, evader_pos, evader_vel = state

        closest_state, closest_time = closest_approach((pursuer_pos, pursuer_vel, evader_pos, evader_vel), 300)
        print(closest_time)
        print(closest_state)
        if closest_time <= 0.1 or 80 <= closest_time:
            pursuer_pos = np.array(pursuer_pos)
            evader_pos = np.array(evader_pos)
            acc_dir = (evader_pos - pursuer_pos)/np.linalg.norm(pursuer_pos - evader_pos)
            final_acc_dir = 0.7 * acc_dir
        elif 25 <= closest_time:
            pos1, _, pos2, _ = closest_state
            closest_approach_dir = (pos2 - pos1)/np.linalg.norm(pos1 - pos2)
            final_acc_dir = 0.7 * closest_approach_dir
        elif closest_time <= 25:
            pos1, vel1, pos2, vel2 = closest_state
            closest_approach_dir = (pos2 - pos1)/np.linalg.norm(pos1 - pos2)
            relative_vel_dir = (vel1 - vel2)/np.linalg.norm(vel1 - vel2)
            final_acc_dir = 0.4 * closest_approach_dir - 0.8 * relative_vel_dir
        return {
            "burn_vec": final_acc_dir.tolist() + [5.0],
            "ref_frame": 1
        }


if __name__ == "__main__":
    my_agent = DeterministicAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=240,
        debug=False)
    runner.run()