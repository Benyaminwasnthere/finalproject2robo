import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse

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

def interpolate(start, goal, resolution=0.1):
    # Interpolate between start and goal configurations
    path = []
    for t in np.arange(0, 1, resolution):
        interpolated_config = (1 - t) * np.array(start) + t * np.array(goal)
        path.append(tuple(interpolated_config))
    path.append(tuple(goal))  # Ensure the goal is included
    return path

def plot_robot(configuration):
    # Plot the robot given a configuration
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(0, SCENE_X)
    ax.set_ylim(0, SCENE_Y)

    center_x, center_y, theta = configuration
    rect_vertices = create_rectangle(center_x, center_y, theta)
    rect = patches.Polygon(rect_vertices, closed=True,fill=False, edgecolor='b')
    ax.add_patch(rect)

    plt.draw()
    plt.pause(0.01)

if __name__ == "__main__":
    print("--start 0.5 0.5 0 --goal 1.2 1.0 0.5")
    parser = argparse.ArgumentParser(description="Rigid Body Motion Planning")
    parser.add_argument("--start", nargs=3, type=float, help="Start configuration")
    parser.add_argument("--goal", nargs=3, type=float, help="Goal configuration")
    args = parser.parse_args()

    start_config = tuple(args.start)
    goal_config = tuple(args.goal)

    path = interpolate(start_config, goal_config, resolution=STEP)

    for config in path:
        plot_robot(config)

    plt.show()


