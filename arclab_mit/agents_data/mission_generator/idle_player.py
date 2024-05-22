import datetime
import time

from arclab_mit.agents_data.mission_generator import generator
from arclab_mit.agents import navball_agent


def idle_play():
    while True:

        runner = None

        # Generate a new mission
        generator.main()

        try:
            # Run the navball agent
            runner = navball_agent.execute_agent()

            # Sleep for a certain amount of time before running again
            print(runner.runner_timeout)
        except:
            print("Navball agent failed to run")
            if runner:
                # Kill the navball agent if not timed out
                navball_agent.kill(runner)


if __name__ == "__main__":
    idle_play()
