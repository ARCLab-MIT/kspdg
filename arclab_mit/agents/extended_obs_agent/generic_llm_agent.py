from threading import Thread
from kspdg.agent_api.base_agent import KSPDGBaseAgent
import openai
import json
import time

import csv
import math
from datetime import datetime

import os
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ.get('OPEN_API_KEY')

class LLMAgent(KSPDGBaseAgent):
    system_prompt_path = None

    def __init__(self, log_metrics=False, log_metrics_filename=None, log_positions=False, log_positions_filename=None):
        super().__init__()
        with open(os.path.join(os.path.dirname(__file__), self.system_prompt_path), 'r') as file:
            system_prompt = file.read()
        self.messages = [{"role": "system", "content": system_prompt}]
        self.prev_time = -100
        self.saved_action = None
        self.refresh_action_duration = 5
        self.updating_action = False
        
        self.log_metrics = log_metrics
        self.log_positions = log_positions
        self.initial_fuel = None
        if self.log_metrics:
            self.log_metrics_filename = log_metrics_filename
            open(self.log_metrics_filename, 'w').close()
        if self.log_positions:
            self.log_positions_filename = log_positions_filename
            open(self.log_positions_filename, 'w').close()
    
    def get_message(self, observation):
        pass
    
    # return None if no manual response; otherwiwse return the response
    def get_manual_response(self, observation):
        pass

    def new_action(self, observation):
        print("=" * 60)
        # print(observation[0])
        user_message = self.get_message(observation)
        self.messages.append({"role": "user", "content": user_message})
        
        print(user_message)
        print("=" * 30)
        
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

        manual_response = self.get_manual_response(observation)
        if manual_response:
            response_message = manual_response
            print("using manual response")
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
        
        print(response_message["content"])

        self.messages.append(response_message)
        
        # getting results of the function call and returning the response
        if response_message.get("function_call"):
            def get_action(throttle):
                return {
                    "burn_vec": throttle + [0.4], # 0.4 doesn't really matter. the smaller the better, but too small might be buggy
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
                print("throttles:", function_response["burn_vec"][0:3])
                return function_response
            except:
                print("error occured while parsing arguments")
                return [0,0,0,0.1]

        print("error: LLM did not call function")
        self.messages.append({"role": "user", "content": "Error: you did not call a function. Remember to use apply_throttle at the end of every response."})
        return [0,0,0,0.1]

    def update_action(self, observation):
        self.updating_action = True
        self.saved_action = self.new_action(observation)
        self.updating_action = False

    def get_action(self, observation):
        if not self.initial_fuel:
            self.initial_fuel = observation[2]
        # log
        if self.log_metrics:
            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(observation[3:6], observation[9:12])]))
            rel_speed = math.sqrt(sum([(a - b) ** 2 for a, b in zip(observation[6:9], observation[12:15])]))
            fuel_used = self.initial_fuel - observation[2]
            with open(self.log_metrics_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    observation[0],
                    distance,
                    rel_speed,
                    fuel_used,
                    (0.1*distance)**2.0 + (0.5*rel_speed)**1.5 + (0.1*fuel_used)**1.25 + (0.01*observation[0])**1.0
                    ])
        
        if self.log_positions:
            with open(self.log_positions_filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([observation[0]] + list(observation[3:15]))

        if observation[0] - self.prev_time >= self.refresh_action_duration and not self.updating_action:
            self.prev_time = observation[0]
            Thread(target=lambda: self.update_action(observation), daemon=True).start()
        return self.saved_action