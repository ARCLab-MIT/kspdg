import json
import os
import time
from os.path import join, dirname

import numpy as np
import openai
from dotenv import load_dotenv

from arclab_mit.agents.fine_tuning_agent import LLMAgent

# dotenv_path = join(dirname(__file__), 'arclib_mit', 'agents', '.env')
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

dotenv_path = join(dirname(__file__), 'alex_prompts.txt')
load_dotenv(dotenv_path)

if __name__ == "__main__":
    model = os.environ['MODEL']
    openai.api_key = os.environ["OPENAI_API_KEY"]

    print(f'Model: {model}')

    functions = [{
        "name": "perform_action",
        "description": "Send the given throttles to the spacecraft.",
        "parameters": {
            "type": "object",
            "properties": {
                "ft": {
                    "type": "string",
                    "enum": ["backward", "none", "forward"],
                    "description": "The forward throttle.",
                },
                "rt": {
                    "type": "string",
                    "enum": ["left", "none", "right"],
                    "description": "The right throttle.",
                },
                "dt": {
                    "type": "string",
                    "enum": ["down", "none", "up"],
                    "description": "The down throttle.",
                },
            },
            "required": ["ft", "rt", "dt"],
        },
    }]

    """
    functions = [{
        "name": "perform_action",
        "description": "Send the given throttles to the spacecraft.",
        "parameters": {
            "type": "object",
            "properties": {
                "ft": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": 1,
                    "description": "The forward throttle.",
                },
                "rt": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": 1,
                    "description": "The right throttle.",
                },
                "dt": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": 1,
                    "description": "The down throttle.",
                },
            },
            "required": ["ft", "rt", "dt"],
        },
    }]
    """

    system_prompt = os.environ["PE_SYSTEM_PROMPT"]
    user_prompt = os.environ["PE_USER_PROMPT"]
    cot = os.environ["PE_CHAIN_OF_THOUGHT"]
    use_cot = (os.environ['USE_COT'].lower() == "true")

    print("System prompt: " + system_prompt)
    print("\n")

    agent = LLMAgent()

    while True:
        observations = input("Observations: ")
        obs = observations
        data = json.loads(obs)
        pursuer_position = np.array([data["pursuer_pos_x"], data["pursuer_pos_y"], data["pursuer_pos_z"]])
        evader_position = np.array([data["evader_pos_x"], data["evader_pos_y"], data["evader_pos_z"]])
        pursuer_velocity = np.array([data["pursuer_vel_x"], data["pursuer_vel_y"], data["pursuer_vel_z"]])
        evader_velocity = np.array([data["evader_vel_x"], data["evader_vel_y"], data["evader_vel_z"]])
        rel_position = evader_position - pursuer_position
        rel_velocity = evader_velocity - pursuer_velocity
        distance = np.linalg.norm(rel_position, ord=2)
        velocity = np.linalg.norm(rel_velocity, ord=2)
        approaching = np.dot(rel_position, rel_velocity) < 0
        print("approaching: " + str(approaching))

        if True:
            txt = user_prompt.format(obs=observations, CoT=cot)
            print ("user prompt:" + txt)
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": txt}
            ]

            time_before = time.time()
            if use_cot:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    functions=functions,
                    #            max_tokens = 150, # limit output tokens (enough for valid responses)
                    temperature=0  # randomness, cool approach if we want to adjust some param with this
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    functions=functions,
        #            max_tokens = 150, # limit output tokens (enough for valid responses)
                    temperature=0  # randomness, cool approach if we want to adjust some param with this
                )
            time_after = time.time()
            print("Chat completion took " + str(time_after - time_before) + " seconds")
            print("approaching: " + str(approaching))

            print(response)
            print(f'Distance: {distance:.2f}, Velocity: {velocity:.2f}, Approaching: {approaching}')
            print('Relative position: ', str(rel_position))
            """
            if response.choices[0].message["content"] is not None:
                content = response.choices[0].message["content"]
                function_args = agent.clean_response(content)
                function_args = json.loads(function_args)
                print("arguments:" + str(function_args))
#                print("Response content:\n" + response.choices[0].message["content"])
            """


