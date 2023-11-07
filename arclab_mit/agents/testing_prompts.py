import openai
import os


def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key == None:
        api_key = input("Please enter your OPENAI_API_KEY: ")
        if not api_key:
            raise AssertionError("Please set the OPENAI_API_KEY variable")
    return api_key


def generate_response(api_key, system_prompt, observation_history):
    prompt = f"{system_prompt}\n{observation_history}\nAgent response:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        api_key=api_key,
    )

    return response.choices[0].text


if __name__ == "__main__":
    my_key = get_api_key()

    # Get the current working directory
    current_directory = os.path.dirname(__file__)

    # Construct the path to the text file
    file_path = os.path.join(current_directory, "prompts", "first_prompt.txt")

    # Open the text file and read its contents
    with open(file_path, "r") as file:
        first_prompt = file.read()

    # p = "given a four by four board,you are at the bottom left corner. Give me a set of moves containing only up, down, left and right to get to the top right of the board in the shortest way possible."
    p = first_prompt
    o = "none"
    print(generate_response(my_key, p, o))
