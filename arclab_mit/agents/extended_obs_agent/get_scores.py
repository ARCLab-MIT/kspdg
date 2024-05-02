import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def get_distances(folder_path):
    best_distances = []
    for file in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                best_distance = min(float(row[1]) for row in reader)
                best_distances.append(best_distance)
        # print(file_path, best_distance)
    return best_distances

# Assuming the provided folder path is correct and accessible
for folder_path in [
    "arclab_mit/agents/extended_obs_agent/baseline_results/llm/e1_i3",
    "arclab_mit/agents/extended_obs_agent/baseline_results/llm/e2_i3",
    "arclab_mit/agents/extended_obs_agent/baseline_results/llm/e1_i3",
    "arclab_mit/agents/extended_obs_agent/baseline_results/llm/e4_i3",
    ]:
    distances = get_distances(folder_path)
    # print(distances)
    distances_pruned = [d for d in distances if d < 600]
    print(len(distances) - len(distances_pruned))
    print(np.mean(distances_pruned))
    print(np.min(distances))

"""
269.2223330006859, 50.0%
295.08697521262684, 30.0%
328.1995776938981, 30.0%
329.7838454179931, 40.0%
"""