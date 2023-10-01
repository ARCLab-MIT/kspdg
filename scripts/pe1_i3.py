# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
This script is a "Hello World" for writing agents that can interact with
a KSPDG environment.

Instructions to Run:
- Start KSP game application.
- Select Start Game > Play Missions > Community Created > pe1_i3 > Continue
- In kRPC dialog box click Add server. Select Show advanced settings and select Auto-accept new clients. Then select Start Server
- In a terminal, run this script

"""
import os
import openai
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role" : "user", "content" : prompt},
                {"role" : "system" , "content": "You are an intelligent agent that runs a satellite in the game Kerbal Space Program, and you are trying to intercept another satellite.\
                                                You are given a lists of numbers, giving the observations of the mission with this structure:\
                                                    [0] : mission elapsed time [s]\
                                                    [1] : current vehicle (pursuer) mass [kg]\
                                                    [2] : current vehicle (pursuer) propellant  (mono prop) [kg]\
                                                    [3:6] : pursuer position wrt CB in right-hand CBCI coords [m]\
                                                    [6:9] : pursuer velocity wrt CB in right-hand CBCI coords [m/s]\
                                                    [9:12] : evader position wrt CB in right-hand CBCI coords [m]\
                                                    [12:15] : evader velocity wrt CB in right-hand CBCI coords [m/s]\
                                                    The way you can control the pursuer is by setting three throttles from 0 to 1 and the duration of the burn, please give me a list with all 4 numbers"}]
    response = openai.Completion.create(
        model=model,
        messages=messages,
        temperature=0,    # randomness, cool approach if we want to adjust some param with this
    )
    return response.choices[0].message["content"]

# instantiate and reset the environment to populate game
env = PE1_E1_I3_Env()
env.reset()

# Environment automatically orients pursuer toward target
# therefore a niave pusuit policy to to simply burn at full
# thrust in pursuer's body-forward direction.
# Do this until the episode
# (Do you think it can intercept even a non-maneuvering evader??)
is_done = False
act = [1.0, 0, 0, 1.0]  # forward throttle, right throttle, down throttle, duration [s]
while not is_done:
    obs, rew, is_done, info = env.step(act)
    response = get_completion(str(obs))
    print(response)
    print("obs: ", str(obs) + "\nrew: ", str(rew) + "\nis_done: ", str(is_done) + "\ninfo: ", str(info) + "\n\n ------------------ \n")

# printout info to see evaluation metrics of episode
print(info)

# close the environments to cleanup any processes
env.close()
