import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

SCENE_X = 2.0
SCENE_Y = 2.0
ROBO_LEN = SCENE_X * 0.1
ROBO_HEIGHT = SCENE_Y * 0.05
STEP = 0.1  # Movement step size
THETA_STEP = 10  # Degrees to rotate the rectangle with each key press
fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, SCENE_X)
ax.set_ylim(0, SCENE_Y)
# Initialize the rotation angle of the robot
theta = 0

def create_rectangle(center_x, center_y, theta):
    # Define the rectangle vertices relative to center
    rect = [(-ROBO_LEN / 2, -ROBO_HEIGHT / 2), (-ROBO_LEN / 2, ROBO_HEIGHT / 2),
            (ROBO_LEN / 2, ROBO_HEIGHT / 2), (ROBO_LEN / 2, -ROBO_HEIGHT / 2)]
    # Convert rotation angle to radians
    theta_rad = theta
    # Create rotation matrix for 2D space
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad), np.cos(theta_rad)]])
    # Rotate and translate vertices
    return [tuple(np.dot(rotation_matrix, np.array([vx, vy])) + np.array([center_x, center_y])) for vx, vy in rect]

def calculate_distance(target, config, alpha=0.7):
    dt = np.sqrt((target[0] - config[0])**2 + (target[1] - config[1])**2)
    dr = np.abs(target[2] - config[2])
    return alpha * dt + (1 - alpha) * dr

def find_nearest_neighbors(target, configs):
    # Calculate the distances between the target and each configuration
    distances = [calculate_distance(target, config) for config in configs]

    # Get the indices sorted by distances in ascending order
    sorted_indices = sorted(range(len(configs)), key=lambda i: distances[i])

    # Return all sorted indices
    return sorted_indices

def plot_robot_configuration(ax, configuration, color, label):
    poly = patches.Polygon(configuration, edgecolor=color, fill=False, label=label)
    ax.add_patch(poly)

if __name__ == "__main__":
    # file = input("New file name, end it with .npy: ")
    file = "rigid_configs.npy"
    target_x = float(input("Enter Target x value: "))
    target_y = float(input("Enter Target y value: "))
    target_theta = float(input("Enter Target theta value: "))
    k = int(input("Enter K: "))

    robot = np.load(file, allow_pickle=True)

    arms_no = robot.shape[0]
    print(arms_no)

    robolist = [(robot[i][0], robot[i][1], robot[i][2]) for i in range(arms_no)]
    print (robolist)

    indices = find_nearest_neighbors((target_x, target_y, target_theta), robolist)
    print("Nearest Neighbors Indices:", indices)



    # Plot the target configuration in black
    plot_robot_configuration(ax, create_rectangle(target_x, target_y, target_theta), 'black', 'Target')

    # Plot the nearest neighbors in red, green, and blue
    for i in range(min(k, 3)):
        color = ['red', 'green', 'blue'][i]
        plot_robot_configuration(ax, create_rectangle(*robolist[indices[i]]), color, f'Neighbor {i+1}')

    # If k is greater than 3, plot the remaining neighbors in yellow
    for i in range(3, k):
        color = 'yellow'
        plot_robot_configuration(ax, create_rectangle(*robolist[indices[i]]), color, f'Neighbor {i+1}')






    plt.show()
