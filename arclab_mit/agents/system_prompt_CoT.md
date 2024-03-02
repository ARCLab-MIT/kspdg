PE_CHAIN_OF_THOUGHT_NAVBALL="
USER: Given these observations {\"vehicle_mass\": 6685.0, \"vehicle_propellant\":
1199.9976806640625, \"pursuer_pos_x\": 707714.9233405391, \"pursuer_pos_y\": -248272.96224747336, \"pursuer_pos_z\":
-0.0005507227224457853, \"pursuer_vel_x\": 718.2699440171854, \"pursuer_vel_y\": 2047.6550602366287, \"pursuer_vel_z\":
-8.727880298612636e-06, \"evader_pos_x\": 708577.1048547915, \"evader_pos_y\": -245801.0041154024, \"evader_pos_z\":
-0.42948227236866804, \"evader_vel_x\": 711.1178432623287, \"evader_vel_y\": 2050.1498254215267, \"evader_vel_z\":
0.0035782107557670575, \"prograde\": [0.6553969259127622, -0.3380649604462953, -0.6754013266366493]},
what is the best throttle to capture evader?\n
ASSISTANT: The x-coordinate of prograde is positive, indicating that evader is moving to the right. The y-coordinate
of the prograde is negative, indicating that evader is moving away. The z-coordinate
of prograde is negative, indicating that evader is moving down.
To capture the evader, we should move in the
opposite direction in the x axes, forward in the y axes, and in the opposite direction in the z axes.
This means we should apply
throttles to move left, forward and up. Therefore we should call
perform_action({\"ft\": \"forward\", \"rt\": \"left\", \"dt\": \"up\"}).
Now answer the following question:\n"