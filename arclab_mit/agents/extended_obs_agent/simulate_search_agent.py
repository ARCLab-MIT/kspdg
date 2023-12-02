from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from arclab_mit.agents.extended_obs_agent.simulate import simulate, closest_approach
from arclab_mit.agents.common import obs_to_state, state_to_message

class SimulateSearchAgent(KSPDGBaseAgent):
    """
    Deterministic agent that searches through action space and picks
    best simulated distance

    I was testing this approach to see if it trivializes the problem, but
    it turns out this approach doesn't really work
    """
    def __init__(self):
        super().__init__()

    def get_action(self, observation):
        t = 0.2
        actions = []
        for a in np.linspace(-1,1,2):
            for b in np.linspace(-1,1,2):
                for c in np.linspace(-1,1,2):
                    actions.append([a,b,c,t])
        min_closest_approach = float("inf")
        min_action = None
        for action in actions:
            state_estimate = obs_to_state(observation)
            for i in range(3):
                # estimate state by saying that it's as if 0.95 of the acceleration happens immediately
                state_estimate[1][i] += action[i] * t * 0.95
            (r_pursuer, _, r_evader, _), _ = closest_approach(state_estimate, 300)
            distance = np.linalg.norm(np.array(r_pursuer) - np.array(r_evader))
            if distance < min_closest_approach:
                min_action = action
                min_closest_approach = distance
            print(action, distance)
        
        print(min_closest_approach)
        print("=" * 30)

        return min_action


if __name__ == "__main__":
    runner = AgentEnvRunner(
        agent=SimulateSearchAgent(), 
        env_cls=PE1_E1_I3_Env, 
        env_kwargs=None,
        runner_timeout=500,
        debug=False
        )
    runner.run()
