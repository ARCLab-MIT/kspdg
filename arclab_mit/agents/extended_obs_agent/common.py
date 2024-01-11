import numpy as np

def round_arr(arr, digits):
    return tuple(round(x,digits) for x in arr.tolist())

# pos/vel must be np arrays
def single_spacecraft_message(pos, vel, name, use_altitude=True, use_velocity=True):
    pos = np.array(pos)
    vel = np.array(vel)
    return "".join([
        f"{name} position (x,y,z): {round_arr(pos, 1)} [m]\n",
        f"{name} altitude: {round(float(np.linalg.norm(pos)), 1)} [m]\n" if use_altitude else "",
        f"{name} velocity (v_x,v_y,v_z): {round_arr(vel, 1)} [m/s]\n" if use_velocity else "",
    ])

# state must be np arrays
def compare_spacecraft_message(state, name1, name2, approach=True, use_velocity=True, extra_label=""):
    state = [np.array(ele) for ele in state]
    pos1, vel1, pos2, vel2 = state
    return "\n".join([
        f"relative position{extra_label} ({name1} position minus {name2} position): {round_arr(pos1 - pos2, 1)} [m]",
        f"distance{extra_label}: {round(float(np.linalg.norm(pos1 - pos2)), 1)} [m]",

        f"direction to accelerate to approach {name2}{extra_label}: {round_arr((pos2 - pos1)/np.linalg.norm(pos1 - pos2), 2)}"
            if approach else f"direction to accelerate to move away from {name2}: {round_arr((pos1 - pos2)/np.linalg.norm(pos1 - pos2), 1)}",
        
        f"relative velocity ({name1} velocity minus {name2} velocity){extra_label}: {round_arr(vel1 - vel2, 2)} [m/s]" if use_velocity else "",
    ])