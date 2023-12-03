from kspdg.agent_api.base_agent import KSPDGBaseAgent
import openai
import json
import time

import os
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ.get('OPEN_API_KEY')

class LLMAgent(KSPDGBaseAgent):
    system_prompt_path = None

    def __init__(self):
        super().__init__()
        with open(os.path.join(os.path.dirname(__file__), self.system_prompt_path), 'r') as file:
            system_prompt = file.read()
        self.messages = [{"role": "system", "content": system_prompt}]
        self.use_manual_response = True
    
    def get_message(self, observation):
        pass

    def get_manual_response(self, observation):
        pass

    def get_action(self, observation):
        # print("=" * 100)
        # print("=" * 100)
        # print("get_action called, prompting ChatGPT...")
        print(observation[0])
        user_message = self.get_message(observation)
        self.messages.append({"role": "user", "content": user_message})
        
        # print(user_message)
        # print("=" * 30)
        
        # repeatedly pop the first pair (index 1 and 2) to not exceed context length
        while True:
            total_len_estimate = 0 # estimate becuase idk how function calls count toward context
            for message in self.messages:
                if message["content"] is not None:
                    total_len_estimate += len(message["content"])
            if total_len_estimate <= 15000: # leaving some buffer for function calling tokens, so not full context length
                break
            self.messages = [self.messages[0]] + self.messages[3:]

        # print(self.messages)

        if self.use_manual_response:
            response_message = self.get_manual_response(observation)
            self.use_manual_response = False
        else:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-3.5-turbo-16k",
                messages=self.messages,
                functions=[{
                    "name": "apply_throttle",
                    "description": "Move the pursuer spacecraft with a (x,y,z) throttle.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "throttle": {
                                "type": "array",
                                "items": {
                                    "type": "number",
                                    "minimum": -1.0,
                                    "maximum": 1.0
                                },
                                "description": "An array of three floats, each between -1.0 and 1.0, that specify the (x,y,z) throttle values."
                            }
                        },
                        "required": ["throttle"],
                    },
                }],
                function_call="auto",
            )
            print("Time taken (seconds): ", time.time() - start_time)
            response_message = response["choices"][0]["message"]
        
        # print(response_message)

        self.messages.append(response_message)
        
        # getting results of the function call and returning the response
        if response_message.get("function_call"):
            duration = 5.0
            def get_action(throttle):
                return {
                    "burn_vec": throttle + [duration],
                    "ref_frame": 1 # celestial ref frame
                }
            available_functions = {
                "apply_throttle": get_action,
            }
            function_name = response_message["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                self.messages.append({"role": "user", "content": "Error: you called the wrong function. Please only use apply_throttle, not python"})
                return [0,0,0,0.1]
            function_to_call = available_functions[function_name]

            try:
                function_args = json.loads(response_message["function_call"]["arguments"])
                function_response = function_to_call(**function_args)
                return function_response
            except:
                print("error occured while parsing arguments")
                return [0,0,0,0.1]

        print("error: LLM did not call function")
        self.messages.append({"role": "user", "content": "Error: you did not call a function. Remember to use apply_throttle at the end of every response."})
        return [0,0,0,0.1]