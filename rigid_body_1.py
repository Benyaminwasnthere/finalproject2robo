import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, transforms

# Importing necessary functions from a custom module for collision checking and scene printing
from collision_checking import print_scene, check_polygons

# Define constants for the scene size, the robot dimensions, and movement steps
SCENE_X = 2.0
SCENE_Y = 2.0
ROBO_LEN = SCENE_X * 0.1
ROBO_HEIGHT = SCENE_Y * 0.05
STEP = 0.1  # Movement step size
THETA_STEP = 10  # Degrees to rotate the rectangle with each key press

# Initialize the rotation angle of the robot
theta = 0

# Function to create and rotate a rectangle around its center
def create_rectangle(center_x, center_y, theta):
    # Define the rectangle vertices relative to center
    rect = [(-ROBO_LEN / 2, -ROBO_HEIGHT / 2), (-ROBO_LEN / 2, ROBO_HEIGHT / 2),
            (ROBO_LEN / 2, ROBO_HEIGHT / 2), (ROBO_LEN / 2, -ROBO_HEIGHT / 2)]
    # Convert rotation angle to radians
    theta_rad = np.radians(theta)
    # Create rotation matrix for 2D space
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad), np.cos(theta_rad)]])
    # Rotate and translate vertices
    return [tuple(np.dot(rotation_matrix, np.array([vx, vy])) + np.array([center_x, center_y])) for vx, vy in rect]

# Function to check for collisions between the robot and obstacles
def check_arm(arm_vert):
    # Loop through each obstacle to check for collisions
    for i in range(polygons.shape[0]):
        if check_polygons(arm_vert, polygons[i]):
            return True
    return False

# Set up the initial plotting environment
fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')
ax.set_xlim(0, SCENE_X)
ax.set_ylim(0, SCENE_Y)

# Load the obstacle polygons from a file
print("rigid_polygons.npy for ta to copy and paste")
file = input("file name: ")

polygons = np.load(file, allow_pickle=True)

# Print the initial scene with obstacles
print_scene(polygons, ax)

# Function to check if the starting position collides with any obstacles
def is_position_clear(start_pos):
    arm_vert = create_rectangle(*start_pos, theta)
    return not check_arm(arm_vert)

# Starting position of the robot, checking for collisions
START = (np.random.uniform(0, SCENE_X), np.random.uniform(0, SCENE_Y))
while not is_position_clear(START):
    # If there's a collision, randomly find a new starting position within the scene bounds
    START = (np.random.uniform(0, SCENE_X), np.random.uniform(0, SCENE_Y))

# Create the rectangle representing the robot with its initial position and color
rectangle = patches.Rectangle((START[0] - ROBO_LEN / 2, START[1] - ROBO_HEIGHT / 2),
                              ROBO_LEN, ROBO_HEIGHT, fill=False, edgecolor='blue')
# Add the rectangle to the plot
ax.add_patch(rectangle)

# Event handler for key presses to control robot movement and rotation
def on_key(event):
    global theta, START, rectangle
    # Check if a relevant key is pressed
    if event.key in ['up', 'down', 'left', 'right']:
        x, y = START
        dx, dy = 0, 0
        dtheta = 0

        # Determine the direction of movement or rotation based on the key
        if event.key == 'up':  # Move forward
            dx = STEP * np.cos(np.radians(theta))
            dy = STEP * np.sin(np.radians(theta))
        elif event.key == 'down':  # Move backward
            dx = -STEP * np.cos(np.radians(theta))
            dy = -STEP * np.sin(np.radians(theta))
        elif event.key == 'right':  # Rotate counter-clockwise
            dtheta = -THETA_STEP
        elif event.key == 'left':  # Rotate clockwise
            dtheta = THETA_STEP

        # Update the rotation angle
        theta += dtheta
        # Update the starting position
        START = (x + dx, y + dy)
        # Get the new vertices after movement and rotation
        arm_vert = create_rectangle(x + dx, y + dy, theta)

        # Only move and rotate the robot if there's no collision
        if not check_arm(arm_vert):
            # Set the new position of the robot
            rectangle.set_x(x - ROBO_LEN / 2 + dx)
            rectangle.set_y(y - ROBO_HEIGHT / 2 + dy)

            # Calculate and apply the cumulative rotation transformation
            rotate_transform = transforms.Affine2D().rotate_deg_around(START[0], START[1], theta)
            rectangle.set_transform(rotate_transform + ax.transData)

            # Redraw the plot with the updated robot position and rotation
            plt.draw()

# Connect the key press event to the handler function
fig.canvas.mpl_connect('key_press_event', on_key)

# Display the plot
plt.show()