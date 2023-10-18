import openai
import argparse


class TextBasedAgent:
    def __init__(self):
        self.api_key = "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"
        self.system_prompt = ""  # TODO add prompt
        self.observation_history = ""  # TODO add observations

    def generate_response(self):
        prompt = f"{self.system_prompt}\n{self.observation_history}\nAgent response:"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,
            api_key=self.api_key,
        )

        return response.choices[0].text

    def parse_response(self, response_text):
        action_vector = [
            float(val) if val.replace(".", "", 1).isdigit() else val
            for val in response_text.split()
        ]
        return action_vector


if __name__ == "__main__":
    text_agent = TextBasedAgent()

    # Generate a response based on the system prompt and observation history
    response_text = text_agent.generate_response()

    # Parse the response to get the action vector
    action_vector = text_agent.parse_response(response_text)

    print("Generated Response:", response_text)
    print("Action Vector:", action_vector)
