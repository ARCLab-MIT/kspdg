def obs_to_state(obs):
    return obs[3:6], obs[6:9], obs[9:12], obs[12:15]

def round_tuple(tup, digits):
    return tuple(round(x,digits) for x in tup)

def state_to_message(state):
    return "\n".join([
        f"pursuer position (x,y,z): {round_tuple(state[0], 1)} [m]",
        f"pursuer velocity (v_x,v_y,v_z): {round_tuple(state[1], 1)} [m/s]",
        f"evader position (x,y,z): {round_tuple(state[2], 1)} [m]",
        f"evader velocity (v_x,v_y,v_z): {round_tuple(state[3], 1)} [m/s]",
        f"relative position (evader position minus pursuer position): {round_tuple(tuple(a - b for a, b in zip(state[2], state[0])), 1)} [m]",
        f"relative velocity (evader velocity minus pursuer velocity): {round_tuple(tuple(a - b for a, b in zip(state[3], state[1])), 1)} [m/s]",
    ])