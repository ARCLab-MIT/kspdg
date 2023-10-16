"""install pynput"""
from pynput import keyboard
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner

class KeyboardControlledAgent(KSPDGBaseAgent):
    """
    Controls:
    H/N: Forward/Backward
    J/L: Left/Right
    I/K: Down/Up
    (controls are taken from here: https://wiki.kerbalspaceprogram.com/wiki/Key_bindings)

    There might be a lag between controls and the game

    Disable the controls in game, or press the controls on another focused window
    """
    def __init__(self):
        super().__init__()
        self.forward_throttle = 0
        self.right_throttle = 0
        self.down_throttle = 0
        listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        listener.start()

    def on_key_press(self, key):
        try:
            print("key pressed:", key.char)
            if key.char == 'h':
                self.forward_throttle = 1.0
            elif key.char == 'n':
                self.forward_throttle = -1.0
            elif key.char == 'j':
                self.right_throttle = -1.0
            elif key.char == 'l':
                self.right_throttle = 1.0
            elif key.char == 'i':
                self.down_throttle = 1.0
            elif key.char == 'k':
                self.down_throttle = -1.0
        except AttributeError:
            pass

    def on_key_release(self, key):
        try:
            print("key released:", key.char)
            if key.char in ('h', 'n'):
                self.forward_throttle = 0
            elif key.char in ('j', 'l'):
                self.right_throttle = 0
            elif key.char in ('i', 'k'):
                self.down_throttle = 0
        except AttributeError:
            pass

    def get_action(self, observation):
        """ compute agent's action given observation
        This function is necessary to define as it overrides 
        an abstract method
        """

        # Return action list
        print(self.forward_throttle, self.right_throttle, self.down_throttle)
        return [self.forward_throttle, self.right_throttle, self.down_throttle, 0.5]

if __name__ == "__main__":
    keyboard_agent = KeyboardControlledAgent()    
    runner = AgentEnvRunner(
        agent=keyboard_agent, 
        env_cls=PE1_E1_I3_Env, 
        env_kwargs=None,
        runner_timeout=300,
        # debug=True
        debug=False
        )
    runner.run()
