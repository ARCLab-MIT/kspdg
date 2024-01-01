import openai
import os

# Set your API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"

# Create an instance of the OpenAIAPI class
client = openai


# Create a new assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview",
)


# Create a new conversation thread
thread = client.beta.threads.create()
print(thread)

# User message to the assistant
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account.",
)

run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

messages = client.beta.threads.messages.list(thread_id=thread.id)

for message in messages.data:
    print(message.role + " : " + message.content[0].text.value)
