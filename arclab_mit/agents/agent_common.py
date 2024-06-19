import math
import os

import numpy as np
from dotenv import load_dotenv

from kspdg.pe1.e1_envs import PE1_E1_I1_Env, PE1_E1_I3_Env, PE1_E1_I4_Env
from kspdg.pe1.e2_envs import PE1_E2_I3_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env, PE1_E3_I4_Env
from kspdg.sb1.e1_envs import SB1_E1_I1_Env, SB1_E1_I2_Env, SB1_E1_I3_Env, SB1_E1_I4_Env, SB1_E1_I5_Env

DEFAULT_TARGET_VIEWING_DISTANCE = 100.0  # [m]
DEFAULT_REWARD_DECAY_COEF = 1e-5  # [1/m^2]


class State:
    """
    The State class holds the observations given by KSP Differential Games environment:

    -   Mission time [s]
    -   Vehicle propellant [kg]
    -   Pursuer position [m]
    -   Pursuer velocity [m/s]
    -   Evader position [m]
    -   Evader velocity [m/s]
    -   Guard position [m] (only for LBG problem)
    -   Guard velocity [m] (only for LBG problem)
    -   Sun position [m] (only for SB problem)

    All these observations except for the sun's position are given in the get_action() call of the
    agent API. Sun's position is directly obtained from the KRPC server.

    All coordinates are given in the right-handed orbital reference frame of the celestial body defined as:
    -   x-axis in an arbitrary direction through the equator (Zenith)
    -   y-axis points in the orbital retrograde direction (West)
    -   z-axis points in the orbital normal direction (North)

    NOTE: Kerbal Space Program uses left-handed coordinate systems and the y-axis points in the orbital
    prograde direction (East). This is why the State class uses the functions State.lh_to_rh() and State.rh_to_lh()
    to convert between left-handed and right-handed coordinate systems.

    This class uses two configuration parameters to generate the JSON structure of a state:
        - use_relative_coordinates: to choose between relative and absolute positions and velocities
        - use_short_names: to choose between short of full argument names. Short names reduce token consumption
          in OpenAI calls.
    """
    DEFAULT_USE_RELATIVE_COORDINATES = False
    DEFAULT_USE_SHORT_NAMES = False
    DEFAULT_USE_PROGRADE = False
    DEFAULT_USE_COT = False
    DEFAULT_USE_COT_SPEED_LIMIT = False

    ROTATION_THRESHOLD = 0.0
    VESSEL_ACCELERATION = 0.2  # [m/s^2]

    def __init__(self, observation, vessel_up=None, sun_position=None):
        self.mission_time = observation[0]
        self.vehicle_mass = observation[1]
        self.vehicle_propellant = observation[2]

        self.pursuer_position = np.array([observation[3], observation[4], observation[5]])
        self.pursuer_velocity = np.array([observation[6], observation[7], observation[8]])

        self.evader_position = np.array([observation[9], observation[10], observation[11]])
        self.evader_velocity = np.array([observation[12], observation[13], observation[14]])

        if len(observation) > 15:
            self.guard_position = np.array([observation[15], observation[16], observation[17]])
            self.guard_velocity = np.array([observation[18], observation[19], observation[20]])

        self.rel_position = self.evader_position - self.pursuer_position
        self.rel_velocity = self.evader_velocity - self.pursuer_velocity

        self.distance = np.linalg.norm(self.rel_position, ord=2)
        self.velocity = np.linalg.norm(self.rel_velocity, ord=2)
        self.time_to_intercept = self.distance / self.velocity

        self.vessel_up = vessel_up

        if self.vessel_up is not None:
            """ Obtain rotation matrix to convert from orbital to surface reference frame of the celestial body
                The surface frame is given by:
                -   x-axis points in the Zenith (e.g. from the center of the celestial body towards the vessel)
                -   y-axis points North and tangential to the surface of the celestial body
                -   z-axis points eastwards and tangential to the surface of the celestial body
            """
            x_axis = self.pursuer_position / np.linalg.norm(self.pursuer_position)
            x_axis = State.rh_to_lh(x_axis)
            y_axis = np.cross(x_axis, [0, 1, 0])
            if np.linalg.norm(y_axis) == 0:
                y_axis = np.cross(x_axis, [0, 0, 1])
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            # Rotation matrix from surface to celestial
            surface_rot_matrix = np.array([x_axis, y_axis, z_axis])
            # Rotation from celestial to surface
            self.surface_rot_matrix = np.linalg.inv(surface_rot_matrix)

            """ Obtain rotation matrix to convert from orbital to vessel reference frame
                The vessel frame is given by:
                -   x-axis points to the right of the vessel
                -   y-axis points at target
                -   z-axis points downwards
            """
            y_axis = self.rel_position / np.linalg.norm(self.rel_position)
            y_axis = State.rh_to_lh(y_axis)
            z_axis = self.vessel_up / np.linalg.norm(self.vessel_up)
            z_axis = State.rh_to_lh(z_axis)
            x_axis = np.cross(y_axis, z_axis)
            # Rotation matrix from vessel to celestial
            vessel_rot_matrix = np.array([x_axis, y_axis, z_axis])
            # Rotation from celestial to vessel
            self.vessel_rot_matrix = np.linalg.inv(vessel_rot_matrix)

            """ Check celestial to vessel transformation
            
                The angle between the relative position and velocity vectors in the vessel frame should be the same as
                the angle between vessel nose and the retrograde vector (as defined by Navball) in the celestial frame.
            """
            # Check vessel transformation
            angle1 = State.angle_between_vectors(self.rel_position, self.rel_velocity)
            angle2 = State.angle_between_vectors([0, 1, 0], self.get_retrograde())
            if abs((angle1 - angle2)) > 1:
                print(f"\x1b[33;20mWarning: angles in celestial {angle1:.2f} and vessel {angle2:.2f} frame differ by {angle1 - angle2:.2f} degrees\x1b[0m")

        self.sun_position = sun_position
        if sun_position is not None:
            p_e_pos = self.evader_position - self.pursuer_position
            p_s_pos = self.sun_position - self.pursuer_position
            angle_cosine = np.dot(p_e_pos, p_s_pos) / (np.linalg.norm(p_e_pos) * np.linalg.norm(p_s_pos))
            self.alignment_angle = math.degrees(np.arccos(angle_cosine))
        else:
            self.alignment_angle = 0

        self.approaching = np.dot(self.rel_position, self.rel_velocity) < 0

        self.distance_to_stop = self.velocity ** 2 / (2 * State.VESSEL_ACCELERATION)

    """ Convert from left-handed to right-handed coordinates
    """
    def lh_to_rh(v):
        # Just revert the sign of the y-coordinate
        return np.array([v[0], -v[1], v[2]])

    """ Convert from right-handed to left-handed coordinates
    """
    def rh_to_lh(v):
        # Just revert the sign of the y-coordinate
        return np.array([v[0], -v[1], v[2]])

    """ Get distance between pursuer and evader.
    """
    def distance_to_target(self):
        return self.distance

    """ Get evader velocity relative to pursuer.
    """
    def velocity_to_target(self):
        return self.velocity

    """ Get angle between vectors in degrees.
    """
    def angle_between_vectors(u, v):
        dp = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
        return math.acos(dp) * 180 / math.pi

    """ Get pitch, heading and roll of given orientation v and vessel up vector. These are the rotation
        angles in degrees around the lateral (pitch), vertical (heading) and longitudinal (roll) axis.
 
        NOTE:
        North, East and Zenith directions in surface frame are given by (0, 1, 0), (0, 0, 1) and (1, 0, 0). Thus:
        -   heading is the angle between v and North
        -   pitch is the angle between v and its horizontal projection (latitude)
        -   roll is the angle between vessel_up and the plane defined by v and Zenith (longitude)
        Applying up-down thrust will primarily affect pitch whereas applying left-right thrust will
        primarily affect roll.
    """
    def get_pitch_heading_roll(v, vessel_up):
        # Returns pitch, heading and roll of v (e.g. rotation around lateral, vertical and longitudinal axis)
        # vessel_up is the up direction of the vessel in surface frame

        # Project v onto horizontal plane
        v_h = (0, v[1], v[2])

        # Compute the pitch (angle between v and horizontal plane)
        pitch = State.angle_between_vectors(v, v_h)
        if v[0] < 0:
            pitch = -pitch

        # Compute the heading (angle between v_h and north)
        north = (0, 1, 0)
        heading = State.angle_between_vectors(v_h, north)
        if v_h[2] < 0:
            heading = 360 - heading

        # Compute the roll
        up = (1, 0, 0)
        plane_normal = np.cross(v, up)
        roll = State.angle_between_vectors(vessel_up, plane_normal)
        # Adjust so that the angle is between -180 and 180 and
        # rolling right is +ve and left is -ve
        if vessel_up[0] > 0:
            roll = -roll
        elif roll < 0:
            roll += 180
        else:
            roll -= 180

        return pitch, heading, roll

    """ Get euler angles (pitch, heading, roll) of lever marker (relative position from pursuer to evader)
    """
    def get_lever(self, ref_frame="vessel", angles=False):
        if self.vessel_up is not None:
            v = State.rh_to_lh(self.evader_position - self.pursuer_position)
            vessel_up = State.rh_to_lh(self.vessel_up)

            # Transform directions to surface reference frame
            if ref_frame == "surface":
                v = np.matmul(v, self.surface_rot_matrix)
                vessel_up = np.matmul(vessel_up, self.surface_rot_matrix)
            else:
                v = np.matmul(v, self.vessel_rot_matrix)
                vessel_up = [0, 0, 1]

            if angles:
                return State.get_pitch_heading_roll(v, vessel_up)
            else:
                return v / np.linalg.norm(v)
        return None

    """ Get pitch, heading and roll of prograde (orientation of vessel's velocity relative to target as defined in KSP)

        NOTE: It is more common to refer to prograde as the relative velocity of the target with respect to the vessel.
    """
    def get_prograde(self, ref_frame="vessel", angles=False):
        if self.vessel_up is not None:
            v = State.rh_to_lh(self.pursuer_velocity - self.evader_velocity)
            vessel_up = State.rh_to_lh(self.vessel_up)

            # Transform directions to surface reference frame
            if ref_frame == "surface":
                v = np.matmul(v, self.surface_rot_matrix)
                vessel_up = np.matmul(vessel_up, self.surface_rot_matrix)
            else:
                v = np.matmul(v, self.vessel_rot_matrix)
                vessel_up = [0, 0, 1]

            if angles:
                return State.get_pitch_heading_roll(v, vessel_up)
            else:
                return v / np.linalg.norm(v)
        return None

    """ Get pitch, heading and roll of retrograde (orientation of target's velocity relative to vessel)
    """
    def get_retrograde(self, ref_frame="vessel", angles=False):
        if self.vessel_up is not None:
            v = State.rh_to_lh(self.evader_velocity - self.pursuer_velocity)
            vessel_up = State.rh_to_lh(self.vessel_up)

            # Transform directions to surface reference frame
            if ref_frame == "surface":
                v = np.matmul(v, self.surface_rot_matrix)
                vessel_up = np.matmul(vessel_up, self.surface_rot_matrix)
            else:
                v = np.matmul(v, self.vessel_rot_matrix)
                vessel_up = [0, 0, 1]

            if angles:
                return State.get_pitch_heading_roll(v, vessel_up)
            else:
                return v / np.linalg.norm(v)
        return None

    """ Compute the reward obtained for this state (only for SB problem).
    """
    def get_reward(self, reward_decay_coef=DEFAULT_REWARD_DECAY_COEF, target_viewing_distance=DEFAULT_TARGET_VIEWING_DISTANCE):
        # get evader position, distance, and unit vector relative to pursuer
        p_vesE_vesP__lhpbody = self.evader_position - self.pursuer_position
        d_vesE_vesP = np.linalg.norm(p_vesE_vesP__lhpbody)
        u_vesE_vesP__lhpbody = p_vesE_vesP__lhpbody / d_vesE_vesP

        # get sun unit vector relative to pursuer
        p_sun_vesP__lhpbody = self.sun_position - self.pursuer_position
        d_sun_vesP = np.linalg.norm(p_sun_vesP__lhpbody)
        u_sun_vesP__lhpbody = p_sun_vesP__lhpbody / d_sun_vesP

        # compute reward. See sb_objective_plot.py for intuition
        # about reward surface shape
        rew = -np.dot(u_vesE_vesP__lhpbody, u_sun_vesP__lhpbody)
        rew *= np.exp(-reward_decay_coef * (d_vesE_vesP - target_viewing_distance) ** 2)

    """ Returns a JSON representation for the state. This representation depends on:
        - Use of relative position / velocities
        - Use of short names for the keys
    """
    def to_json(self, scenario, use_relative_coordinates=False, use_short_names=False, use_prograde=False,
                use_cot=False, use_cot_speed_limit=False):
        scenario = scenario.lower()
        state_json = {
            "vehicle_mass": self.vehicle_mass,
            "vehicle_propellant": self.vehicle_propellant,
    
            "pursuer_pos_x": self.pursuer_position[0],
            "pursuer_pos_y": self.pursuer_position[1],
            "pursuer_pos_z": self.pursuer_position[2],
    
            "pursuer_vel_x": self.pursuer_velocity[0],
            "pursuer_vel_y": self.pursuer_velocity[1],
            "pursuer_vel_z": self.pursuer_velocity[2],
    
            "evader_pos_x": self.evader_position[0],
            "evader_pos_y": self.evader_position[1],
            "evader_pos_z": self.evader_position[2],
    
            "evader_vel_x": self.evader_velocity[0],
            "evader_vel_y": self.evader_velocity[1],
            "evader_vel_z": self.evader_velocity[2],
        }
        if scenario.startswith('sb'):
            # Replace names and add sun position
            state_json["vessel_pos"] = [state_json.pop('pursuer_pos_x'),
                                        state_json.pop('pursuer_pos_y'),
                                        state_json.pop('pursuer_pos_z')]
    
            state_json["vessel_vel"] = [state_json.pop('pursuer_vel_x'),
                                        state_json.pop('pursuer_vel_y'),
                                        state_json.pop('pursuer_vel_z')]

            state_json["satellite_pos"] = [state_json.pop('evader_pos_x'),
                                           state_json.pop('evader_pos_y'),
                                           state_json.pop('evader_pos_z')]

            state_json["satellite_vel"] = [state_json.pop('evader_vel_x'),
                                           state_json.pop('evader_vel_y'),
                                           state_json.pop('evader_vel_z')]

            state_json["sun_pos"] = self.sun_position

        elif scenario.startswith('lbg'):
            # Replace names
            state_json["bandit_pos"] = [state_json.pop('pursuer_pos_x'),
                                        state_json.pop('pursuer_pos_y'),
                                        state_json.pop('pursuer_pos_z')]

            state_json["bandit_vel"] = [state_json.pop('pursuer_vel_x'),
                                        state_json.pop('pursuer_vel_y'),
                                        state_json.pop('pursuer_vel_z')]

            state_json["lady_pos"] = [state_json.pop('evader_pos_x'),
                                      state_json.pop('evader_pos_y'),
                                      state_json.pop('evader_pos_z')]

            state_json["lady_vel"] = [state_json.pop('evader_vel_x'),
                                      state_json.pop('evader_vel_y'),
                                      state_json.pop('evader_vel_z')]

            state_json["guard_pos"] = [self.guard_position[0],
                                       self.guard_position[1],
                                       self.guard_position[2]]

            state_json["guard_vel"] = [self.guard_velocity[0],
                                       self.guard_velocity[1],
                                       self.guard_velocity[2]]

        if use_relative_coordinates:
            if scenario.startswith('pe'):
                state_json['relative_pos_x'] = state_json.pop('evader_pos_x') - state_json.pop('pursuer_pos_x')
                state_json['relative_pos_y'] = state_json.pop('evader_pos_y') - state_json.pop('pursuer_pos_y')
                state_json['relative_pos_z'] = state_json.pop('evader_pos_z') - state_json.pop('pursuer_pos_z')

                state_json['relative_vel_x'] = state_json.pop('evader_vel_x') - state_json.pop('pursuer_vel_x')
                state_json['relative_vel_y'] = state_json.pop('evader_vel_y') - state_json.pop('pursuer_vel_y')
                state_json['relative_vel_z'] = state_json.pop('evader_vel_z') - state_json.pop('pursuer_vel_z')
            elif scenario.startswith('lbg'):
                state_json["relative_lady_pos"] = (self.evader_position - self.pursuer_position).tolist()
                state_json["relative_lady_vel"] = (self.evader_velocity - self.pursuer_velocity).tolist()
                state_json["relative_guard_pos"] = (self.guard_position - self.pursuer_position).tolist()
                state_json["relative_guard_vel"] = (self.guard_velocity - self.pursuer_velocity).tolist()
                _ = state_json.pop('bandit_pos')
                _ = state_json.pop('bandit_vel')
                _ = state_json.pop('lady_pos')
                _ = state_json.pop('lady_vel')
                _ = state_json.pop('guard_pos')
                _ = state_json.pop('guard_vel')
            elif scenario.startswith('sb'):
                state_json["relative_satellite_pos"] = (self.evader_position - self.pursuer_position).tolist()
                state_json["relative_satellite_vel"] = (self.evader_velocity - self.pursuer_velocity).tolist()
                _ = state_json.pop('satellite_pos')
                _ = state_json.pop('satellite_vel')
                _ = state_json.pop('vessel_pos')
                _ = state_json.pop('vessel_vel')
                state_json['relative_sun_pos'] = (state_json.pop('sun_pos') - self.pursuer_position).tolist()

        if use_short_names:
            state_json["m"] = state_json.pop("vehicle_mass")
            state_json["f"] = state_json.pop("vehicle_propellant")
            if scenario.startswith('pe'):
                if use_relative_coordinates:
                    state_json["rp"] = state_json.pop("relative_satellite_pos")
                    state_json["rv"] = state_json.pop("relative_satellite_vel")
                    state_json["rsunp"] = state_json.pop("relative_sun_pos")
                else:
                    state_json["vp"] = state_json.pop("vessel_pos")
                    state_json["vv"] = state_json.pop("vessel_vel")
                    state_json["sp"] = state_json.pop("satellite_pos")
                    state_json["sv"] = state_json.pop("satellite_vel")
                    state_json["sunp"] = state_json.pop("sun_pos")
            elif scenario.startswith('pe'):
                if use_relative_coordinates:
                    state_json["rx"] = state_json.pop("relative_pos_x")
                    state_json["ry"] = state_json.pop("relative_pos_y")
                    state_json["rz"] = state_json.pop("relative_pos_z")
                    state_json["rvx"] = state_json.pop("relative_vel_x")
                    state_json["rvy"] = state_json.pop("relative_vel_y")
                    state_json["rvz"] = state_json.pop("relative_vel_z")
                else:
                    state_json["px"] = state_json.pop("pursuer_pos_x")
                    state_json["py"] = state_json.pop("pursuer_pos_y")
                    state_json["pz"] = state_json.pop("pursuer_pos_z")
                    state_json["pvx"] = state_json.pop("pursuer_vel_x")
                    state_json["pvy"] = state_json.pop("pursuer_vel_y")
                    state_json["pvz"] = state_json.pop("pursuer_vel_z")
                    state_json["ex"] = state_json.pop("evader_pos_x")
                    state_json["ey"] = state_json.pop("evader_pos_y")
                    state_json["ez"] = state_json.pop("evader_pos_z")
                    state_json["evx"] = state_json.pop("evader_vel_x")
                    state_json["evy"] = state_json.pop("evader_vel_y")
                    state_json["evz"] = state_json.pop("evader_vel_z")
            """
            # To be completed
            elif scenario.startswith('lbg'):
            """

        """ Add prograde direction
        
            NOTE: need to revert direction since LLM considers prograde as the relative velocity of evader
            whereas KSP considers prograde as the relative velocity of pursuer.
        """
        if use_prograde:
            prograde = self.get_prograde()
            rotation = max(abs(prograde[0]), abs(prograde[2]))
            if rotation < State.ROTATION_THRESHOLD:
                """ Discard x and z components of prograde """
                prograde = np.array([0, 1, 0]) if prograde[1] > 0 else np.array([0, -1, 0])
            state_json["prograde"] = prograde.tolist()

            """ Add relative velocity
            """
            if use_cot_speed_limit:
                approach_velocity = self.velocity
                if not self.approaching:
                    approach_velocity = 0
                state_json['approach_velocity'] = approach_velocity

        # Ensure sun_pos goes last in the dictionary
        if scenario.startswith('sb'):
            if use_relative_coordinates:
                if use_short_names:
                    state_json["rsunp"] = state_json.pop("rsunp")
                else:
                    state_json["relative_sun_pos"] = state_json.pop("relative_sun_pos")
            else:
                if use_short_names:
                    state_json["sunp"] = state_json.pop("sunp")
                else:
                    state_json["sun_pos"] = state_json.pop("sun_pos")

        return state_json

    """ Evaluates the alignment angle and the distance for the state. Possible evaluations are:
    "excellent", "good", "average", "poor" and "extremely poor".
    """
    def evaluate_angle_distance(self):
        angle = np.abs(self.alignment_angle)
        distance = self.distance

        if angle < 90:
            angle_gauge = "extremely poor"
        elif angle < 160:
            angle_gauge = "poor"
        elif angle < 170:
            angle_gauge = "average"
        elif angle < 175:
            angle_gauge = "good"
        else:
            angle_gauge = "excellent"

        if distance < 50:
            distance_gauge = "good"
        elif distance < 150:
            distance_gauge = "excellent"
        elif distance < 200:
            distance_gauge = "good"
        elif distance < 500:
            distance_gauge = "average"
        elif distance < 1000:
            distance_gauge = "poor"
        else:
            distance_gauge = "extremely poor"

        return angle_gauge, distance_gauge

    def show(self):
        print(f'Distance: {self.distance:.2f}')
        print(f'Velocity: {self.velocity:.2f}')
        print(f'Mission time: {self.mission_time:.2f}')
        print(f'Fuel: {self.vehicle_propellant:.2f}')
        print('Sun location: ' + str(self.sun_position))
        print(f'Alignment angle: {self.alignment_angle:.4f}')
        print(f'Is approaching: {self.approaching}')


class Action:
    """
    The Action class holds the throttles of the spacecraft in the order:
        - Forward (1) / Backward (-1)
        - Right (1) / Left (-1)
        - Down (1) / Up (-1)
    Throttles are given as integer values where 0 means "no throttle".
    NOTE: throttle duration is not currently stored since the agent operates using a constant duration.

    This class uses one configuration parameters to generate the JSON structure of an action:
        - use_enum: to choose between integer (-1, 0, +1) and enumerated values in the throttles.
    """
    DEFAULT_USE_ENUM = True

    def __init__(self, action):
        self.action = action

    """ Translate throttles to enumerated values.
    """
    def to_enum(self):
        result = ["none", "none", "none"]

        if self.action[0] == -1:
            result[0] = 'backward'
        elif self.action[0] == 1:
            result[0] = 'forward'

        if self.action[1] == -1:
            result[1] = 'left'
        elif self.action[1] == 1:
            result[1] = 'right'

        if self.action[2] == -1:
            result[2] = 'down'
        elif self.action[2] == 1:
            result[2] = 'up'

        # result = ["\"" + item + "\"" for item in result]
        return result

    """ Translate throttles from enumerated values
    """
    @staticmethod
    def from_enum(action):
        result = [0, 0, 0, action[3]]

        if action[0] == "backward":
            result[0] = -1
        elif action[0] == "forward":
            result[0] = 1

        if action[1] == "left":
            result[1] = -1
        elif action[1] == "right":
            result[1] = 1

        if action[2] == "down":
            result[2] = -1
        elif action[2] == "up":
            result[2] = 1

        return result

    """ Returns a JSON representation for the action. This representation uses integer or enumerated
    values.
    """
    def to_json(self, use_enum=False):
        action = self.action
        if use_enum:
            action = self.to_enum()
        action_json = {
            "ft": action[0],
            "rt": action[1],
            "dt": action[2]
        }
        return action_json

def set_env_paths():
    base_directory = os.path.dirname(__file__)

    env_path = os.path.join(base_directory, '.env')
    load_dotenv(env_path)

    # Load configuration from alex_prompt.txt
    prompts_path = os.path.join(base_directory, 'alex_prompts.txt')
    load_dotenv(prompts_path)

def setup_scenarios() -> dict:
    scenarios = dict()

    scenarios["PE1_E1_I1"] = PE1_E1_I1_Env
    scenarios["PE1_E1_I3"] = PE1_E1_I3_Env
    scenarios["PE1_E1_I4"] = PE1_E1_I4_Env
    scenarios["PE1_E3_I3"] = PE1_E3_I3_Env
    scenarios["PE1_E3_I4"] = PE1_E3_I4_Env

    scenarios["PE1_E2_I3"] = PE1_E2_I3_Env

    scenarios["SB1_E1_I1"] = SB1_E1_I1_Env
    scenarios["SB1_E1_I2"] = SB1_E1_I2_Env
    scenarios["SB1_E1_I3"] = SB1_E1_I3_Env
    scenarios["SB1_E1_I4"] = SB1_E1_I4_Env
    scenarios["SB1_E1_I5"] = SB1_E1_I5_Env

    return scenarios
