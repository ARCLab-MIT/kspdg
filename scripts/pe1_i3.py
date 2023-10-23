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

openai.api_key = "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" # This is a test key, not working anymore
#os.environ["OPENAI_API_KEY"]

def get_completion_calculator(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt},
                {"role": "system", "content": "You are an intelligent agent that runs a satellite in the game Kerbal Space Program, and you are trying to intercept another satellite.\
                                                You are given a list of three numbers that will be more than sufficient to calculate the intercept\
                                                Please return me four numbers ranging from 0 to 1 that represent the forward, right, and down throttle, and the duration of the burn [s] separated by commas"}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0# randomness, cool approach if we want to adjust some param with this
    )
    return response.choices[0].message["content"]

def get_completion_trainer(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt},
                {"role": "system", "content": "You are a language model trainer that has to pick three of these numbers that are more useful for calculating the spacecraft's throttles\
                                                You are given a list of numbers, giving the observations of the mission with this structure:\
                                                    position 0 : mission elapsed time [s]\
                                                    position 1 : current vehicle (pursuer) mass [kg]\
                                                    position 2 : current vehicle (pursuer) propellant  (mono prop) [kg]\
                                                    position 3 to position 6 : pursuer position wrt CB in right-hand CBCI coords [m]\
                                                    position 6 to position 9 : pursuer velocity wrt CB in right-hand CBCI coords [m/s]\
                                                    position 9 to position 12 : evader position wrt CB in right-hand CBCI coords [m]\
                                                    position 12 to position 15 : evader velocity wrt CB in right-hand CBCI coords [m/s]\
                                                    Give me the three numbers as a prompt for another model"}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0# randomness, cool approach if we want to adjust some param with this
    )
    return response.choices[0].message["content"]

# instantiate and reset the environment to populate game
env = PE1_E1_I3_Env()
env.reset()

# Environment automatically orients pursuer toward target
# therefore a niave pusuit policy to simply burn at full
# thrust in pursuer's body-forward direction.
# Do this until the episode
# (Do you think it can intercept even a non-maneuvering evader??)
is_done = False
act = [1.0, 0, 0, 1.0]  # forward throttle, right throttle, down throttle, duration [s]
while not is_done:
    obs, rew, is_done, info = env.step(act)
    print(obs)
    response = get_completion_trainer(str(obs))
    print(response)
    response = get_completion_calculator(response)
    print(response)
    #forward, right, down, duration = response.split(",")
    #print(forward + " " + right + " " + down + " " + duration)
    print("obs: ", str(obs) + "\nrew: ", str(rew) + "\nis_done: ", str(is_done) + "\ninfo: ", str(info) + "\n\n ------------------ \n")
    #act = [float(forward), float(right), float(down), float(duration)]

# printout info to see evaluation metrics of episode
print(info)

# close the environments to clean up any processes
env.close()
