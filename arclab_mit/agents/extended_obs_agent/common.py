import numpy as np

def round_arr(arr, digits):
    return tuple(round(x,digits) for x in arr.tolist())

# pos/vel must be np arrays
def single_spacecraft_message(pos, vel, name):
    pos = np.array(pos)
    vel = np.array(vel)
    return "\n".join([
        f"{name} position (x,y,z): {round_arr(pos, 1)} [m]",
        f"{name} altitude: {round(float(np.linalg.norm(pos)), 1)} [m]",
        f"{name} velocity (v_x,v_y,v_z): {round_arr(vel, 1)} [m/s]",
    ])

# state must be np arrays
def compare_spacecraft_message(state, name1, name2, approach=True):
    state = [np.array(ele) for ele in state]
    pos1, vel1, pos2, vel2 = state
    return "\n".join([
        f"relative position ({name1} position minus {name2} position): {round_arr(pos1 - pos2, 1)} [m]",
        f"distance: {round(float(np.linalg.norm(pos1 - pos2)), 1)} [m]",

        f"direction to accelerate to approach {name2}: {round_arr((pos2 - pos1)/np.linalg.norm(pos1 - pos2), 1)}"
            if approach else f"direction to accelerate to move away from {name2}: {round_arr((pos1 - pos2)/np.linalg.norm(pos1 - pos2), 1)}",
        
        f"relative velocity ({name1} velocity minus {name2} velocity): {round_arr(vel1 - vel2, 1)} [m/s]",
    ])