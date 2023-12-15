from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
import openai
import json

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

openai.api_key = "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"

# os.environ.get('OPENAI_API_KEY')


class LLMAgent(KSPDGBaseAgent):
    """An agent that uses ChatGPT to make decisions based on observations."""

    def __init__(self, **kwargs):
        super().__init__()

    def get_action(self, observation):
        """Compute the agent's action given the observation."""

        # Calculate the distance between the pursuer and the evader
        distance_to_evader = (
            (observation[9] - observation[3]) ** 2
            + (observation[10] - observation[4]) ** 2
            + (observation[11] - observation[5]) ** 2
        ) ** 0.5

        # If the distance is greater than 1000 meters, prompt ChatGPT for action
        user_message = "\n".join(
            [
                f"mission elapsed time: {observation[0]} [s]",
                f"current vehicle (pursuer) mass: {observation[1]} [kg]",
                f"current vehicle (pursuer) propellant (mono prop): {observation[2]} [kg]",
                f"pursuer position x: {observation[3]} [m]",
                f"pursuer position y: {observation[4]} [m]",
                f"pursuer position z: {observation[5]} [m]",
                f"pursuer velocity x: {observation[6]} [m/s]",
                f"pursuer velocity y: {observation[7]} [m/s]",
                f"pursuer velocity z: {observation[8]} [m/s]",
                f"evader position x: {observation[9]} [m]",
                f"evader position y: {observation[10]} [m]",
                f"evader position z: {observation[11]} [m]",
                f"evader velocity x: {observation[12]} [m/s]",
                f"evader velocity y: {observation[13]} [m/s]",
                f"evader velocity z: {observation[14]} [m/s]",
            ]
        )

        messages = [
            {
                "role": "system",
                "content": "You are controlling a spacecraft agent in Kerbal Space Program. Your job is to control a pursuit spacecraft to intercept an evading spacecraft. You will be given observations detailing your position, your speed, the other spacecraft's position, and the other spacecraft's speed. You should reason through how to proceed and call the perform_action function with the throttle values. Only use this function.",
            },
            {"role": "user", "content": user_message},
        ]
        functions = [
            {
                "name": "perform_action",
                "description": "Send the given throttles to the spacecraft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "forward_throttle": {
                            "type": "integer",
                            "description": "The forward throttle.",
                        },
                        "right_throttle": {
                            "type": "integer",
                            "description": "The right throttle.",
                        },
                        "down_throttle": {
                            "type": "integer",
                            "description": "The down throttle.",
                        },
                    },
                    "required": [
                        "forward_throttle",
                        "right_throttle",
                        "down_throttle",
                    ],
                },
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto",
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            duration = 10.0
            available_functions = {
                "perform_action": lambda forward_throttle, right_throttle, down_throttle: [
                    forward_throttle,
                    right_throttle,
                    down_throttle,
                    duration,
                ],
            }
            function_name = response_message["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                return [0, 0, 0, 0.1]

            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(**function_args)
            return function_response

        # Calculate the vector from pursuer to evader
        evader_position = [observation[9], observation[10], observation[11]]
        pursuer_position = [observation[3], observation[4], observation[5]]
        direction_vector = [evader_position[i] - pursuer_position[i] for i in range(3)]

        # Normalize the direction vector
        length = (
            direction_vector[0] ** 2
            + direction_vector[1] ** 2
            + direction_vector[2] ** 2
        ) ** 0.5
        normalized_direction = [direction_vector[i] / length for i in range(3)]

        # Calculate dot products
        forward_dot = sum(normalized_direction[i] * [1, 0, 0][i] for i in range(3))
        right_dot = sum(normalized_direction[i] * [0, 1, 0][i] for i in range(3))
        down_dot = sum(normalized_direction[i] * [0, 0, 1][i] for i in range(3))

        # Map directions to throttle values (you can adjust these mappings)
        forward_throttle = max(0, forward_dot)
        right_throttle = max(0, right_dot)
        down_throttle = max(0, down_dot)

        # Ensure that the sum of squares of throttle values does not exceed 1
        magnitude = (
            forward_throttle**2 + right_throttle**2 + down_throttle**2
        ) ** 0.5
        if magnitude > 1:
            forward_throttle /= magnitude
            right_throttle /= magnitude
            down_throttle /= magnitude

        small_adjustments = [forward_throttle, right_throttle, down_throttle, 0.1]
        return small_adjustments


if __name__ == "__main__":
    my_agent = LLMAgent()
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=240,
        debug=False,
    )
    runner.run()
