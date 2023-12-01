class SlidingWindow:

    """A sliding window class with the history of user prompts and assistant responses."""

    """ Size is the number of messages to keep in the window."""
    def __init__(self, size: int = 4):
        # Message window
        self.size = size
        self.window = []

    """ Return the window length."""
    def len(self):
        return len(self.window)

    """ Return the list will all messages in the window """
    def get_messages(self):
        messages = []
        for item in self.window:
            for message in item:
                messages.append(message)
        return messages

    """ Add prompt / response pair to the window. If the window is full, remove the oldest message.
        Arguments is a dictionary of arguments passed to perform_action function.
        """
    def append(self, prompt, arguments):
        """ Append unless window size is 0 """
        if self.size <= 0:
            return

        """ If the window is full, remove the oldest message."""
        if len(self.window) >= self.size:
            self.window.pop(0)

        """ Append user/assistant messages to the window."""
        messages = [{"role": "user", "content": prompt}]
        messages.append({"role": "assistant", "content": None, "function_call": {"name" : "perform_action", "arguments": arguments}})
        self.window.append(messages)
