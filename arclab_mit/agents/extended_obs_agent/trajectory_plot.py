import pandas as pd
import matplotlib.pyplot as plt
import os

filename = "positions_2024-03-30_12-24-12.csv"
df_csv = pd.read_csv(filename, header=None, sep=',')

# Create a new DataFrame with the desired format
df_csv_new = pd.DataFrame({
    'pursuer_pos': df_csv.iloc[:, 1:4].apply(lambda x: [float(val) for val in x], axis=1),
    'evader_pos': df_csv.iloc[:, 7:10].apply(lambda x: [float(val) for val in x], axis=1)
})

# Use the new DataFrame for plotting
df_csv = df_csv_new


figure = plt.figure()
ax = plt.axes(projection='3d')
ax.tick_params(axis='x', labelsize=6, rotation=45, pad=10)
for label in ax.get_xticklabels():
    label.set_horizontalalignment('center')
    label.set_verticalalignment('bottom')
ax.tick_params(axis='y', labelsize=6, rotation=-15, pad=5)
for label in ax.get_yticklabels():
    label.set_horizontalalignment('center')
    label.set_verticalalignment('bottom')
ax.tick_params(axis='z', labelsize=6, pad=0)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

#            ax.invert_yaxis()

x = [p[0] for p in df_csv['pursuer_pos']]
y = [p[1] for p in df_csv['pursuer_pos']]
z = [p[2] for p in df_csv['pursuer_pos']]
print(x, y, z)
ax.plot3D(x, y, z, 'green', label='pursuer')
ax.scatter(x[-1],y[-1],z[-1], '-', c="green")
x = [p[0] for p in df_csv['evader_pos']]
y = [p[1] for p in df_csv['evader_pos']]
z = [p[2] for p in df_csv['evader_pos']]
ax.plot3D(x, y, z, 'red', label='evader')
ax.scatter(x[-1], y[-1], z[-1], '-', c="red")
#            ax.scatter(x, y, z, c="red")
ax.set_zlim(zmin=-50, zmax=50)
"""
elev = 80
azim = 45
roll = 45
ax.view_init(elev, azim, roll)
"""
plt.legend(loc="upper right")
plt.xlim(700000, 750000)
plt.ylim(-250000, -100000)
figure.align_labels()
figure.savefig("fig_3D_pursuer_evader_series_" + os.path.basename(filename) + ".png", format='png')
plt.close(figure)

""" Plot celestial body pursuer and evader positions time series chart (2D Chart)
"""
figure = plt.figure()
ax = plt.axes()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.invert_yaxis()
x = [p[0] for p in df_csv['pursuer_pos']]
y = [p[1] for p in df_csv['pursuer_pos']]
ax.plot(x, y, 'green', label='pursuer')
#            ax.scatter(x[-1], y[-1], '-', c="green")
x = [p[0] for p in df_csv['evader_pos']]
y = [p[1] for p in df_csv['evader_pos']]
ax.plot(x, y, 'red', label='evader')
#            ax.scatter(x[-1], y[-1], '-', c="red")
plt.legend(loc="upper right")
figure.savefig("fig_2D_pursuer_evader_series_" + os.path.basename(filename) + ".png", format='png')
plt.close(figure)