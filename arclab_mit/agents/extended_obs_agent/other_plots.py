import matplotlib.pyplot as plt
import os
import pandas as pd

filename = "positions_2024-03-30_12-24-12.csv"

# Specify the column names
column_names = ['t', 'x1', 'y1', 'z1', 'vx1', 'vy1', 'vz1', 'x2', 'y2', 'z2', 'vx2', 'vy2', 'vz2']

# Read the CSV file without headers and assign column names
df_csv = pd.read_csv(filename, header=None, names=column_names, nrows=217)

# Rename the columns to match the expected names in the function
df_csv = df_csv.rename(columns={"t": "time"})

# Calculate the relative positions and velocities
df_csv['rel_pos'] = list(zip(df_csv['x2'] - df_csv['x1'], df_csv['y2'] - df_csv['y1'], df_csv['z2'] - df_csv['z1']))
df_csv['rel_vel'] = list(zip(df_csv['vx2'] - df_csv['vx1'], df_csv['vy2'] - df_csv['vy1'], df_csv['vz2'] - df_csv['vz1']))

# Calculate the distance and velocity magnitudes
df_csv['distance'] = ((df_csv['x2'] - df_csv['x1'])**2 + (df_csv['y2'] - df_csv['y1'])**2 + (df_csv['z2'] - df_csv['z1'])**2)**0.5
df_csv['velocity'] = ((df_csv['vx2'] - df_csv['vx1'])**2 + (df_csv['vy2'] - df_csv['vy1'])**2 + (df_csv['vz2'] - df_csv['vz1'])**2)**0.5


best_distance = df_csv.min()['distance']
figure, axs = plt.subplots(4, constrained_layout=True)
axs[0].tick_params(axis='x', which='major', labelsize=8)
axs[0].tick_params(axis='y', which='major', labelsize=8)
l1, = axs[0].plot(df_csv['time'],  [p[0] for p in df_csv['rel_pos']], c='green', label='position')
axs[0].set_xlabel('Time (s)', fontsize=8)
axs[0].set_ylabel(r'$\Delta$x (m)', fontsize=8)
#            axs[0].set_yscale('symlog')
axs[0].grid(linestyle='--')
axs_speed = axs[0].twinx()
axs_speed.set_ylim(-30, 5)

axs_speed.tick_params(axis='y', which='major', labelsize=8)
l2, = axs_speed.plot(df_csv['time'],  [p[0] for p in df_csv['rel_vel']], c='blue', label='velocity')
axs_speed.set_ylabel(r'$\Delta$vx (m/s)', fontsize=8)
axs_speed.grid(linestyle='--')
plt.legend([l1, l2], ["position", "velocity"], loc="upper right", fontsize=8)
axs[1].tick_params(axis='x', which='major', labelsize=8)
axs[1].tick_params(axis='y', which='major', labelsize=8)
l1, = axs[1].plot(df_csv['time'],  [p[1] for p in df_csv['rel_pos']], c='green', label='position')
axs[1].set_xlabel('Time (s)', fontsize=8)
axs[1].set_ylabel(r'$\Delta$y (m)', fontsize=8)
#            axs[1].set_yscale('symlog')
axs[1].grid(linestyle='--')
axs_speed = axs[1].twinx()

axs_speed.set_ylim(-80, 5)

axs_speed.tick_params(axis='y', which='major', labelsize=8)
l2, = axs_speed.plot(df_csv['time'],  [p[1] for p in df_csv['rel_vel']], c='blue', label='velocity')
axs_speed.set_ylabel(r'$\Delta$vy (m/s)', fontsize=8)
axs_speed.grid(linestyle='--')
plt.legend([l1, l2], ["position", "velocity"], loc="upper right", fontsize=8)
axs[2].tick_params(axis='x', which='major', labelsize=8)
axs[2].tick_params(axis='y', which='major', labelsize=8)
l1, = axs[2].plot(df_csv['time'],  [p[2] for p in df_csv['rel_pos']], c='green', label='position')
axs[2].set_xlabel('Time (s)', fontsize=8)
axs[2].set_ylabel(r'$\Delta$z (m)', fontsize=8)
#            axs[2].set_yscale('symlog')
axs[2].grid(linestyle='--')
axs_speed = axs[2].twinx()

axs_speed.set_ylim(-1.5, 1.8)

axs_speed.tick_params(axis='y', which='major', labelsize=8)
l2, = axs_speed.plot(df_csv['time'],  [p[2] for p in df_csv['rel_vel']], c='blue', label='velocity')
axs_speed.set_ylabel(r'$\Delta$vz (m/s)', fontsize=8)
axs_speed.grid(linestyle='--')
plt.legend([l1, l2], ["position", "velocity"], loc="upper right", fontsize=8)
axs[3].tick_params(axis='x', which='major', labelsize=8)
axs[3].tick_params(axis='y', which='major', labelsize=8)
l1, = axs[3].plot(df_csv['time'],  df_csv['distance'], c='green', label='position')
axs[3].set_xlabel('Time (s)', fontsize=8)
axs[3].set_ylabel('Distance (m)', fontsize=8)
axs[3].grid(linestyle='--')
axs_speed = axs[3].twinx()

axs_speed.set_ylim(0, 80)

axs_speed.tick_params(axis='y', which='major', labelsize=8)
l2, = axs_speed.plot(df_csv['time'],  df_csv['velocity'], c='blue', label='speed')
axs_speed.set_ylabel(r'Speed (m/s)', fontsize=8)
axs_speed.grid(linestyle='--')

axs[0].set_ylim(-100, 900)
axs[1].set_ylim(0, 2600)
axs[2].set_ylim(-10, 10)
axs[3].set_ylim(0, 2700)

plt.legend([l1, l2], ["distance", "speed"], loc="upper right", fontsize=8)
figure.savefig("fig_rel_pos_series_" + os.path.basename(filename) + ".png", format='png')
plt.close(figure)