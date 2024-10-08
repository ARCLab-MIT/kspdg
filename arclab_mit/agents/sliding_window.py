import json
import numpy as np

from arclab_mit.agents.agent_common import Action



class SlidingWindow:
    """
    The SlidingWindow class holds the last N conversations of the agent where N is the size of a
    sliding window. Each conversation contains the action generated by the agent given some state observations.

    This class uses the following configuration parameters to generate the JSON structure of a state:
        - size: the size of the sliding window.
        - embed_history: to choose between placing the conversation history in user/assistant pairs or the
          embed it in the user prompt.

    The implementation of this class is adapted to the message structure of the OpenAI API.
    """

    DEFAULT_SLIDING_WINDOW_SIZE = 0
    DEFAULT_EMBED_HISTORY = True

    # State/action pair form oldest to newest
    _history = []

    def __init__(self, size: int = DEFAULT_SLIDING_WINDOW_SIZE, scenario="PE",
                 use_relative_coordinates=False, use_short_names=False, use_enum=True,
                 use_prograde_marker=False, use_cot=False, use_cot_speed_limit=False,
                 embed_history=False,
                 system_prompt="", user_prompt="", cot_prompt=None, assistant_content="",
                 history_prompt="", history_item_prompt=""):

        # Window size
        self.size = size

        # KSDPG scenario
        self.scenario = scenario

        # State configuration parameters
        self.use_relative_coordinates = use_relative_coordinates
        self.use_short_names = use_short_names
        self.use_enum = use_enum
        self.use_prograde_marker = use_prograde_marker
        self.use_cot = use_cot
        self.use_cot_speed_limit = use_cot_speed_limit

        # Sliding window configuration parameters
        self.embed_history = embed_history
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.assistant_content = assistant_content
        self.cot_prompt = cot_prompt
        self.history_prompt = history_prompt
        self.history_item_prompt = history_item_prompt

    """
    Adds a state / action pair. Default action None means that action
    is not known at the time of the call.    
    """
    def add(self, state, action=None):

        # Initialize window with given state and default action [0,0,0]
        if len(self._history) == 0:
            for i in range(self.size):
                padded_action = Action([0, 0, 0])
                self._history.append({"state": state,
                                     "action": padded_action})

        # Add state / action pair
        self._history.append({"state": state,
                             "action": action})

    """
    Removes and returns the last state / action pair.
    """
    def pop(self):
        return self._history.pop()

    """
    Replaces the action at selected pos. Negative values of pos mean position starting
    from the end.
    """
    def set_action(self, pos, action):
        self._history[pos]["action"] = action

    """
    Returns the message structure of the conversation at position pos in the sliding window
    excluding the system prompt. This structure uses the format of the OpenAI API.
    
    Negative values of pos refer to positions starting from the end of the conversation history
    (e.g. -1 refers to the last conversation).
    """
    def get_message_structure(self, pos):
        state = self._history[pos]["state"]
        action = self._history[pos]["action"]

        """ Generate Chain-of-Thought if necessary
        """
        chain_of_thought = ""
        calculations = ""
        if self.cot_prompt is not None:
            if self.scenario.lower().startswith('pe'):
                chain_of_thought = self.cot_prompt
                calculations = f", where relative position is {state.distance:.2f}[m] and relative velocity is {state.velocity:.2f}[m/s]"
            elif self.scenario.lower().startswith('sb'):
                angle_gauge, distance_gauge = state.evaluate_angle_distance()
                chain_of_thought = self.cot_prompt.format(angle_gauge=angle_gauge,
                                                          angle=state.alignment_angle,
                                                          distance_gauge=distance_gauge,
                                                          distance=state.distance)

        user_prompt = self.user_prompt.format(obs=json.dumps(state.to_json(self.scenario,
                                                                           self.use_relative_coordinates,
                                                                           self.use_short_names,
                                                                           self.use_prograde_marker,
                                                                           self.use_cot,
                                                                           self.use_cot_speed_limit)),
                                              distance_to_stop=state.distance_to_stop,
                                              calculations=calculations,
                                              CoT=chain_of_thought)
        message_structure = {
            "messages": [{"role": "user", "content": user_prompt}]
        }

        if action is not None:
            action_str = "perform_action(" + json.dumps(action.to_json(self.use_enum)) + ")"
            if self.scenario.lower().startswith('pe'):
                message_structure["messages"] \
                    .append({"role": "assistant",
                             "content": self.assistant_content.format(action=action_str),
                             "function_call": {"name": "perform_action",
                                               "arguments": json.dumps(action.to_json(self.use_enum))}})
            else:
                message_structure["messages"] \
                    .append({"role": "assistant",
                             "content": self.assistant_content.format(action=action_str),
                             "function_call": {"name": "perform_action",
                                               "arguments": json.dumps(action.to_json(self.use_enum))}})
        return message_structure

    """
    Returns the message structure of the last N conversations including the system prompt
    where N is the size of the sliding window. This structure uses the format of the OpenAI API.
    """
    def get_messages(self):
        """ Insert the system prompt
        """
        messages = []
        if self.system_prompt != "":
            messages.append({"role": "system", "content": self.system_prompt})

        if self.embed_history:
            """ Past N conversations are included in the user prompt
            """
            user_prompt = ""
            if self.size > 0:
                history_msg = []
                for i in range(-(self.size+1), -1):
                    item = self._history[i]
                    state = item["state"]
                    action = item["action"]
                    if self.cot_prompt is None:
                        history_msg.append(self.history_item_prompt
                                           .format(obs=json.dumps(state.to_json(self.scenario,
                                                                                self.use_relative_coordinates,
                                                                                self.use_short_names,
                                                                                self.use_prograde_marker))))
                    else:
                        angle_gauge, distance_gauge = state.evaluate_angle_distance()
                        history_msg.append(self.history_item_prompt
                                           .format(obs=json.dumps(state.to_json(self.scenario,
                                                                                self.use_relative_coordinates,
                                                                                self.use_short_names,
                                                                                self.use_prograde_marker)),
                                                   angle_gauge=angle_gauge, angle=state.alignment_angle,
                                                   distance_gauge=distance_gauge, distance=state.distance,
                                                   action=json.dumps(action.to_json(self.use_enum))))
                user_prompt = self.history_prompt + '\n'.join(history_msg) + '\n'
            message_structure = self.get_message_structure(-1)
            message_structure["messages"][0]["content"] = \
                user_prompt + message_structure["messages"][0]["content"]
            messages['messages'] += message_structure["messages"]
        else:
            """ Past N conversations are included before the user prompt
            """
            for i in range(-(self.size+1), 0):
                message_structure = self.get_message_structure(i)
                messages += message_structure["messages"]

        return messages
