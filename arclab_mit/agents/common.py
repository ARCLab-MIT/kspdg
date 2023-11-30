import numpy as np

def obs_to_state(obs):
    return obs[3:6], obs[6:9], obs[9:12], obs[12:15]

def round_arr(arr, digits):
    return tuple(round(x,digits) for x in arr.tolist())

def state_to_message(state):
    state = [np.array(ele) for ele in state]
    return "\n".join([
        f"pursuer position (x,y,z): {round_arr(state[0], 1)} [m]",
        f"pursuer altitude: {round(float(np.linalg.norm(state[0])), 1)} [m]",
        f"pursuer velocity (v_x,v_y,v_z): {round_arr(state[1], 1)} [m/s]",
        f"evader position (x,y,z): {round_arr(state[2], 1)} [m]",
        f"evader altitude: {round(float(np.linalg.norm(state[2])), 1)} [m]",
        f"evader velocity (v_x,v_y,v_z): {round_arr(state[3], 1)} [m/s]",
        f"relative position (pursuer position minus evader position): {round_arr(state[0] - state[2], 1)} [m]",
        f"distance: {round(float(np.linalg.norm(state[0] - state[2])), 1)} [m]",
        f"direction to accelerate to approach evader: {round_arr((state[2] - state[0])/np.linalg.norm(state[0] - state[2]), 1)}",
        f"relative velocity (pursuer velocity minus evader velocity): {round_arr(state[1] - state[3], 1)} [m/s]",
    ])