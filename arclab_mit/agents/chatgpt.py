import os
from dotenv import load_dotenv
import openai
from os.path import join, dirname
import os
import time
from arclab_mit.agents.fine_tuning_agent_history import LLMAgent
import json

# dotenv_path = join(dirname(__file__), 'arclib_mit', 'agents', '.env')
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

dotenv_path = join(dirname(__file__), 'alex_prompts_v2.txt')
load_dotenv(dotenv_path)

if __name__ == "__main__":
    model = os.environ['MODEL']
    openai.api_key = os.environ["OPENAI_API_KEY"]

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

    print("System prompt: " + system_prompt)
    print("\n")

    agent = LLMAgent()

    while True:
        observations = input("Observations: ")

        if True:
            txt = user_prompt.format(obs=observations, CoT=cot)
            print ("user prompt:" + txt)
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": txt}
            ]

            time_before = time.time()
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                 functions=functions,
    #            max_tokens = 150, # limit output tokens (enough for valid responses)
                temperature=0  # randomness, cool approach if we want to adjust some param with this
            )
            time_after = time.time()
            print("Chat completion took " + str(time_after - time_before) + " seconds")

            print(response)
            if response.choices[0].message["content"] is not None:
                content = response.choices[0].message["content"]
                try:
                    print("Response content:\n" + content)
                    function_args = agent.clean_response(content)
                    function_args = json.loads(function_args)
                    print("arguments:" + str(function_args))
                except Exception as ex:
                    print("Error parsing response: " + str(ex))
    #                print("Response content:\n" + response.choices[0].message["content"])


