import os
from anthropic import Anthropic

class ClaudeAPI:
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = Anthropic(api_key=input("Enter API key:"))

    def send_message(self, user_message):
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": user_message}]
        )
        return message.content

# Example Usage
model_id = "claude-3-opus-20240229"  # Replace with your specific model ID
claude_api = ClaudeAPI(model_id)

if __name__ == "__main__":
    wanna_continue = True
    while wanna_continue:
        # For a POST request with a user-provided prompt
        prompt = input("Enter your prompt: ")
        response = claude_api.send_message(prompt)
        print(response)
        wanna_continue = input("Do you want to continue? (y/n) ") == 'y'
