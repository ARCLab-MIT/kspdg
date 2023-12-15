# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
Test that require:
1. KSP to be running
2. Play Mission mode -> lbg_i2
3. kRPC server to be running
"""
import pytest
import time
import numpy as np

import kspdg.utils.constants as C
from kspdg.lbg1.lg0_envs import LBG1_LG0_I2_Env
from kspdg.lbg1.lg1_envs import LBG1_LG1_I2_Env
from kspdg.lbg1.lg2_envs import LBG1_LG2_I2_Env

@pytest.fixture
def lbg1_lg0_i2_env():
    """setup and teardown of the LBG1_LG0_I2_Env object connected to kRPC server"""
    env = LBG1_LG0_I2_Env()
    env.reset()
    yield env
    env.close()

@pytest.fixture
def lbg1_lg1_i2_env():
    """setup and teardown of the LBG1_LG1_I2_Env object connected to kRPC server"""
    env = LBG1_LG1_I2_Env()
    env.reset()
    yield env
    env.close()

@pytest.fixture
def lbg1_lg2_i2_env():
    """setup and teardown of the LBG1_LG2_I2_Env object connected to kRPC server"""
    env = LBG1_LG2_I2_Env()
    env.reset()
    yield env
    env.close()

def test_smoketest_lg0(lbg1_lg0_i2_env):
    """Ensure no errors are thrown from starting LG0 environment"""
    pass

def test_smoketest_lg0(lbg1_lg1_i2_env):
    """Ensure no errors are thrown from starting LG1 environment"""
    pass

def test_smoketest_lg0(lbg1_lg1_i2_env):
    """Ensure no errors are thrown from starting LG2 environment"""
    pass

def test_observation_0(lbg1_lg0_i2_env):
    """Check that observation produces reasonable values"""

    # ~~~ ARRANGE ~~~

    if lbg1_lg0_i2_env is None:
        env = LBG1_LG0_I2_Env()
        env.reset()
    else:
        env = lbg1_lg0_i2_env

    # ~~~ ACT ~~~

    # Wait a little while to let any reset maneuvers to settle
    time.sleep(0.2)

    # query observation
    obs = env.get_observation()

    # ~~~ ASSERT ~~~

    # check consistency of size of observation
    assert len(obs) == env.PARAMS.OBSERVATION.LEN
    assert obs.shape == env.observation_space.shape

    # mission time: startup should take less than 5 seconds
    exp_max_startup = 10.0  # [sec]
    assert obs[0] < exp_max_startup

    # bandit mass
    mass_eps = 10.0 # [kg]
    exp_mass = 6.685e3  # [kg] expected mass of bandit vehicle
    exp_prop_mass = 1.2e3   # [kg] expect propellant mass in bandit
    assert obs[1] > exp_mass - mass_eps
    assert obs[1] < exp_mass + mass_eps
    assert obs[2] > exp_prop_mass - mass_eps
    assert obs[2] < exp_prop_mass + mass_eps

    # bandit position checks
    exp_dist_b_cb = 7.5e5   # [m] expected distance of bandit from center of central body
    dist_b_cb_eps = 1e2     # [m] margin of error on bandit position
    mes_pos_b_cb__rhcbci = obs[3:6] # [m] position of bandit wrt central body expressed in rhcbci coords as measured
    mes_dist_b_cb = np.linalg.norm(mes_pos_b_cb__rhcbci)    # [m] distance of bandit from center of body as measured
    assert mes_dist_b_cb > exp_dist_b_cb - dist_b_cb_eps
    assert mes_dist_b_cb < exp_dist_b_cb + dist_b_cb_eps

    # bandit velocity checks
    exp_spd_b_cb = np.sqrt(C.KERBIN.MU/exp_dist_b_cb)  # [m/s] expected speed of bandit wrt central body (circular orbit)
    spd_b_cb_eps = 10.0 # [m/s] margin of error on bandit speed
    mes_vel_b_cb__rhcbci = obs[6:9] # [m/s] velocity of bandit wrt central body expressed in rhcbci coords as measured
    mes_spd_b_cb = np.linalg.norm(mes_vel_b_cb__rhcbci)  # [m/s] speed of bandit wrt central body as measured
    assert mes_spd_b_cb > exp_spd_b_cb - spd_b_cb_eps
    assert mes_spd_b_cb < exp_spd_b_cb + spd_b_cb_eps

    # lady position checks
    exp_dist_l_cb = 7.5e5   # [m] expected distance of lady from center of central body
    dist_l_cb_eps = 1e2     # [m] margin of error on lady position
    mes_pos_l_cb__rhcbci = obs[9:12]    # [m] position of lady wrt central body expressed in rhcbci coords as measured
    mes_dist_l_cb = np.linalg.norm(mes_pos_l_cb__rhcbci)    # [m] distance of lady from center of body as measured
    assert mes_dist_l_cb > exp_dist_l_cb - dist_l_cb_eps
    assert mes_dist_l_cb < exp_dist_l_cb + dist_l_cb_eps

    # lady velocity checks
    exp_spd_l_cb = np.sqrt(C.KERBIN.MU/exp_dist_l_cb)  # [m/s] expected speed of guard wrt central body (circular orbit)
    spd_l_cb_eps = 10.0 # [m/s] margin of error on guard speed
    mes_vel_l_cb__rhcbci = obs[12:15] # [m/s] velocity of guard wrt central body expressed in rhcbci coords as measured
    mes_spd_l_cb = np.linalg.norm(mes_vel_l_cb__rhcbci)  # [m/s] speed of guard wrt central body as measured
    assert mes_spd_l_cb > exp_spd_l_cb - spd_l_cb_eps
    assert mes_spd_l_cb < exp_spd_l_cb + spd_l_cb_eps

    # guard position checks
    exp_dist_g_cb = 7.5e5   # [m] expected distance of guard from center of central body
    dist_g_cb_eps = 1e2     # [m] margin of error on guard position
    mes_pos_g_cb__rhcbci = obs[15:18]    # [m] position of guard wrt central body expressed in rhcbci coords as measured
    mes_dist_g_cb = np.linalg.norm(mes_pos_g_cb__rhcbci)    # [m] distance of lady from center of body as measured
    assert mes_dist_g_cb > exp_dist_g_cb - dist_g_cb_eps
    assert mes_dist_g_cb < exp_dist_g_cb + dist_g_cb_eps

    # guard velocity checks
    exp_spd_g_cb = np.sqrt(C.KERBIN.MU/exp_dist_g_cb)  # [m/s] expected speed of guard wrt central body (circular orbit)
    spd_g_cb_eps = 10.0 # [m/s] margin of error on guard speed
    mes_vel_g_cb__rhcbci = obs[18:] # [m/s] velocity of guard wrt central body expressed in rhcbci coords as measured
    mes_spd_g_cb = np.linalg.norm(mes_vel_g_cb__rhcbci)  # [m/s] speed of guard wrt central body as measured
    assert mes_spd_g_cb > exp_spd_g_cb - spd_g_cb_eps
    assert mes_spd_g_cb < exp_spd_g_cb + spd_g_cb_eps

def test_convert_rhntw_to_rhvbody_0(lbg1_lg0_i2_env):
    '''check along-track vec in right-hand NTW frame transforms to forward in right-hand pursuer body coords'''

    # rename for ease of use
    env = lbg1_lg0_i2_env
    env.conn.space_center.target_vessel = None
    env.vesBandit.control.rcs = False
    time.sleep(0.5)
    v_exp__rhpbody = [1, 0, 0]

    # vector pointing along track
    v__rhntw = [0, 1, 0]
    env.vesBandit.control.sas_mode = env.vesBandit.control.sas_mode.prograde
    time.sleep(2.0)   # give time to re-orient
    v__rhpbody = lbg1_lg0_i2_env.convert_rhntw_to_rhvbody(v__rhntw, vessel=env.vesBandit)
    assert np.allclose(v__rhpbody, v_exp__rhpbody, atol=1e-2)

    # vector pointing radial out
    v__rhntw = [1, 0, 0]
    env.vesBandit.control.sas_mode = env.vesBandit.control.sas_mode.radial
    time.sleep(2.0)   # give time to re-orient
    v__rhpbody = lbg1_lg0_i2_env.convert_rhntw_to_rhvbody(v__rhntw, vessel=env.vesBandit)
    assert np.allclose(v__rhpbody, v_exp__rhpbody, atol=1e-2)

    # vector pointing normal
    v__rhntw = [0, 0, 1]
    env.vesBandit.control.sas_mode = env.vesBandit.control.sas_mode.normal
    time.sleep(2.0)   # give time to re-orient
    v__rhpbody = lbg1_lg0_i2_env.convert_rhntw_to_rhvbody(v__rhntw, vessel=env.vesBandit)
    assert np.allclose(v__rhpbody, v_exp__rhpbody, atol=1e-2)

    # vector pointing retrograde
    v__rhntw = [0, -1, 0]
    env.vesBandit.control.sas_mode = env.vesBandit.control.sas_mode.retrograde
    time.sleep(2.0)   # give time to re-orient
    v__rhpbody = lbg1_lg0_i2_env.convert_rhntw_to_rhvbody(v__rhntw, vessel=env.vesBandit)
    assert np.allclose(v__rhpbody, v_exp__rhpbody, atol=1e-2)

    # vector pointing in-radial
    v__rhntw = [-1, 0, 0]
    env.vesBandit.control.sas_mode = env.vesBandit.control.sas_mode.anti_radial
    time.sleep(2.0)   # give time to re-orient
    v__rhpbody = lbg1_lg0_i2_env.convert_rhntw_to_rhvbody(v__rhntw, vessel=env.vesBandit)
    assert np.allclose(v__rhpbody, v_exp__rhpbody, atol=1e-2)

    # vector pointing retrograde
    v__rhntw = [0, 0, -1]
    env.vesBandit.control.sas_mode = env.vesBandit.control.sas_mode.anti_normal
    time.sleep(2.0)   # give time to re-orient
    v__rhpbody = lbg1_lg0_i2_env.convert_rhntw_to_rhvbody(v__rhntw, vessel=env.vesBandit)
    assert np.allclose(v__rhpbody, v_exp__rhpbody, atol=1e-2)

def test_step_action_space_0(lbg1_lg0_i2_env):
    '''check that step accepts various action input formats without error'''
    # ~~ ARRANGE ~~

    if lbg1_lg0_i2_env is None:
        env = LBG1_LG0_I2_Env()
        env.reset()
    else:
        env = lbg1_lg0_i2_env

    # ~~ ACT ~~ 

    # Backward compatibility to legacy action space
    env.step([1.0, 0, 0, 0.1])  # throttle in rhpbody

    # Backward compat to v0.3 action space
    env.step({
        "burn_vec":[1.0, 0, 0, 0.1],
        "ref_frame":0   # body coords (rhpbody)
    })
    env.step({
        "burn_vec":[1.0, 0, 0, 0.1],
        "ref_frame":1   # celestial-body-centered inertial (rhcbci)
    })
    env.step({
        "burn_vec":[1.0, 0, 0, 0.1],
        "ref_frame":2   # orbital NTW (rhntw)
    })

    with pytest.raises(ValueError):
        env.step({
            "burn_vec":[1.0, 0, 0, 0.1],
            "ref_frame":3   # invalid
        })

    # v0.4 action space
    env.step({
        "burn_vec":[1.0, 0, 0, 0.1],
        "vec_type":0,   # throttle
        "ref_frame":0   # rhpbody
    })
    env.step({
        "burn_vec":[1000.0, 0, 0, 0.1],
        "vec_type":1,    # thrust
        "ref_frame":0   # rhpbody
    })
    env.step({
        "burn_vec":[1000.0, 0, 0, 0.1],
        "vec_type":1,    # thrust
        "ref_frame":1   # rhcbci
    })
    env.step({
        "burn_vec":[1000.0, 0, 0, 0.1],
        "vec_type":1,    # thrust
        "ref_frame":2   # rhntw
    })

    with pytest.raises(ValueError):
        env.step({
            "burn_vec":[1.0, 0, 0, 0.1],
            "vec_type":2,    # invalid
            "ref_frame":0
        })

def test_step_action_ref_frame_1(lbg1_lg0_i2_env):
    '''check that inclination only burn does not affect speed or radius'''
    # ~~ ARRANGE ~~

    if lbg1_lg0_i2_env is None:
        env = LBG1_LG0_I2_Env()
        env.reset()
    else:
        env = lbg1_lg0_i2_env

    env.conn.space_center.target_vessel = None
    env.vesBandit.control.rcs = False
    env.vesBandit.control.sas_mode = env.vesBandit.control.sas_mode.normal
    time.sleep(2.0)   # give time to re-orient
    env.vesBandit.control.rcs = True

    # ~~ ACT ~~ 
    obs0 = env.get_observation()
    r0 = np.linalg.norm(obs0[3:6])
    v0 = np.linalg.norm(obs0[6:9])
    env.step({
        "burn_vec":[0, 0, 1.0, 1.0],
        "vec_type":0,
        "ref_frame":1
    })
    time.sleep(1.0)

    obs1 = env.get_observation()
    r1 = np.linalg.norm(obs1[3:6])
    v1 = np.linalg.norm(obs1[6:9])

    # ~~ ASSERT ~~
    assert np.isclose(r0, r1)
    assert np.isclose(v0, v1)

def test_step_action_ref_frame_2(lbg1_lg0_i2_env):
    '''action in ref frame 2 produces expected delta-vs

    Works by pointing the vehicle in various directions but then 
    calling a orbit normal burn which should keep radial distance
    and speed the same
    '''
    # ~~ ARRANGE ~~

    if lbg1_lg0_i2_env is None:
        env = LBG1_LG0_I2_Env()
        env.reset()
    else:
        env = lbg1_lg0_i2_env

    env.conn.space_center.target_vessel = None
    time.sleep(0.1)

    # ~~ ACT and ASSERT ~~

    for point_dir in [
        env.vesBandit.control.sas_mode.prograde, 
        env.vesBandit.control.sas_mode.normal,
        env.vesBandit.control.sas_mode.radial]:

        env.vesBandit.control.rcs = False
        env.vesBandit.control.sas_mode = point_dir
        time.sleep(2.0)   # give time to re-orient
        env.vesBandit.control.rcs = True

        obs0 = env.get_observation()
        r0 = np.linalg.norm(obs0[3:6])
        v0 = np.linalg.norm(obs0[6:9])

        env.step({
            "burn_vec":[0, 0, 1.0, 1.0],
            "vec_type":0,
            "ref_frame":2
        })
        time.sleep(1.0)

        obs1 = env.get_observation()
        r1 = np.linalg.norm(obs1[3:6])
        v1 = np.linalg.norm(obs1[6:9])

        # ~~ ASSERT ~~
        assert np.isclose(r0, r1)
        assert np.isclose(v0, v1)

 