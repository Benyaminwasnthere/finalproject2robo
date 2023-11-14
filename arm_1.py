import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import math
from random import uniform, randint
import random
from collision_checking import print_scene, check_polygons

FLAG_START = 0
SCENE_X = 2.00
SCENE_Y = 2.00

START_POINT_X = 1
START_POINT_Y = 1

JOINT_RADIUS = SCENE_X * 0.025
LEN_J1 = SCENE_X * 0.2
LEN_J2 = SCENE_X * 0.125

print("arm_polygons.npy is the name to copy for tas")
file = input("file name: ")
polygons = np.load(file, allow_pickle=True)



fig, ax = plt.subplots(dpi=100)
ax.set_aspect('equal')

# Create a figure and axis


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_rectangle_vertices(reference_point, length, height, angle):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle)

    # Unpack reference point coordinates
    x, y = reference_point

    # Calculate the coordinates of the other three vertices before rotation
    vertices = [
        [x, y],  # Reference point
        [x + length * np.cos(angle_radians), y + length * np.sin(angle_radians)],  # Top right
        [x + length * np.cos(angle_radians) - height * np.sin(angle_radians),
         y + length * np.sin(angle_radians) + height * np.cos(angle_radians)],  # Bottom right
        [x - height * np.sin(angle_radians), y + height * np.cos(angle_radians)]  # Bottom left
    ]

    return vertices


def calculate_angle_between_points(pointA, pointB, pointC):
    vectorAB = np.array(pointA) - np.array(pointB)
    vectorBC = np.array(pointC) - np.array(pointB)

    dot_product = np.dot(vectorAB, vectorBC)
    magnitude_AB = np.linalg.norm(vectorAB)
    magnitude_BC = np.linalg.norm(vectorBC)

    cosine_theta = dot_product / (magnitude_AB * magnitude_BC)

    # Calculate the angle in radians
    theta = np.arccos(cosine_theta)

    # Convert radians to degrees
    degrees = np.degrees(theta)

    return np.deg2rad(degrees)


# Create a figure and axis


# Define joint parameters
joint_radius = JOINT_RADIUS

# Initialize the main joint (fixed)
main_joint_coordinates = np.array([START_POINT_X, START_POINT_Y])
main_joint = patches.Circle(main_joint_coordinates, joint_radius, fill=True, color='black')
ax.add_patch(main_joint)

# Initialize the second joint (rotating around the main joint)
angle_degrees_second_joint = 0
second_joint_radius = LEN_J1

second_joint_coordinates = np.array([main_joint_coordinates[0] + second_joint_radius,
                                     main_joint_coordinates[1]])
second_joint = patches.Circle(second_joint_coordinates, joint_radius, fill=True, color='red')
ax.add_patch(second_joint)

# ------------------------------------
rectangle1 = patches.Rectangle(np.array([START_POINT_X + JOINT_RADIUS, START_POINT_Y - JOINT_RADIUS]),
                               LEN_J1 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, fill=False, edgecolor='blue')
ax.add_patch(rectangle1)

rec1_dist = calculate_distance(main_joint_coordinates[0], main_joint_coordinates[1], rectangle1.get_x(),
                               rectangle1.get_y())

rec1_angle = calculate_angle_between_points(
    [START_POINT_X + JOINT_RADIUS, START_POINT_Y - JOINT_RADIUS],
    [START_POINT_X, START_POINT_Y],
    [START_POINT_X + JOINT_RADIUS, START_POINT_Y]
)

rec1_ver = calculate_rectangle_vertices((START_POINT_X + JOINT_RADIUS, START_POINT_Y - JOINT_RADIUS),
                                        LEN_J1 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, 0)

# ------------------------------------

# Initialize the third joint (spinning around the second joint while moving with it)
angle_degrees_third_joint = 0
third_joint_radius = LEN_J2

third_joint_coordinates = np.array([second_joint_coordinates[0] + third_joint_radius,
                                    second_joint_coordinates[1]])

third_joint = patches.Circle(third_joint_coordinates, joint_radius, fill=True, color='blue')
ax.add_patch(third_joint)

# ------------------------------------
rectangle2 = patches.Rectangle(
    np.array([second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1] - JOINT_RADIUS]),
    LEN_J2 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, fill=False, edgecolor='blue')
ax.add_patch(rectangle2)

rec2_dist = calculate_distance(second_joint_coordinates[0], second_joint_coordinates[1], rectangle2.get_x(),
                               rectangle2.get_y())

rec2_angle = calculate_angle_between_points(
    [second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1] - JOINT_RADIUS],
    [second_joint_coordinates[0], second_joint_coordinates[1]],
    [second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1]]
)

rec2_ver = calculate_rectangle_vertices(
    (second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1] - JOINT_RADIUS),
    LEN_J2 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, 0)

# ------------------------------------

# Set axis limits
plt.xlim(0, SCENE_X)  # Set x-axis limits
plt.ylim(0, SCENE_Y)  # Set y-axis limits

# Set aspect ratio to equal
ax.set_aspect('equal', adjustable='box')


def get_circle(x, y):
    num_sides = 16
    angles = np.linspace(0, 2 * np.pi, num_sides + 1)[:num_sides]
    return [(x + JOINT_RADIUS * np.cos(angle), y + JOINT_RADIUS * np.sin(angle)) for angle in angles]

def check_joint(arm_vert):
    polygon_no = polygons.shape[0]
    for i in range(polygon_no):
        if check_polygons(arm_vert, polygons[i]):
            return True
    return False

# Function to update the positions of the rotating joints and redraw the plot
def update_joint_positions(add_two,add_three):
    global angle_degrees_second_joint, angle_degrees_third_joint

    local_angle_degrees_second_joint = angle_degrees_second_joint + add_two
    local_angle_degrees_third_joint = angle_degrees_third_joint + add_three
    local_angle_degrees_second_joint %= 360
    local_angle_degrees_third_joint %= 360

    # Calculate new position for the second joint in a circular motion around the main joint
    angle_radians_second_joint = np.deg2rad(local_angle_degrees_second_joint)
    new_second_joint_x = main_joint_coordinates[0] + second_joint_radius * np.cos(angle_radians_second_joint)
    new_second_joint_y = main_joint_coordinates[1] + second_joint_radius * np.sin(angle_radians_second_joint)

    # Calculate new position for the third joint in a circular motion around the second joint
    angle_radians_third_joint = np.deg2rad(local_angle_degrees_third_joint)
    new_third_joint_x = new_second_joint_x + third_joint_radius * np.cos(angle_radians_third_joint)
    new_third_joint_y = new_second_joint_y + third_joint_radius * np.sin(angle_radians_third_joint)

    new_rec1_x = main_joint_coordinates[0] + rec1_dist * np.cos(angle_radians_second_joint - rec1_angle)
    new_rec1_y = main_joint_coordinates[1] + rec1_dist * np.sin(angle_radians_second_joint - rec1_angle)

    new_rec2_x = new_second_joint_x + rec2_dist * np.cos(angle_radians_third_joint - rec2_angle)
    new_rec2_y = new_second_joint_y + rec2_dist * np.sin(angle_radians_third_joint - rec2_angle)

    rec1_ver = calculate_rectangle_vertices((new_rec1_x, new_rec1_y),
                                            LEN_J1 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, local_angle_degrees_second_joint)
    rec2_ver = calculate_rectangle_vertices((new_rec2_x, new_rec2_y),
                                            LEN_J2 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, local_angle_degrees_third_joint)

    j1 = get_circle(main_joint_coordinates[0], main_joint_coordinates[1])
    j2 = get_circle(new_second_joint_x, new_second_joint_y)
    j3 = get_circle(new_third_joint_x, new_third_joint_y)







    # calculate collision----------------------------------------------------------------------------------------
    if not check_joint(j1) and not check_joint(j2) and not check_joint(j3) and not check_joint(rec1_ver) and not check_joint(rec2_ver):
        second_joint.set_center((new_second_joint_x, new_second_joint_y))
        third_joint.set_center((new_third_joint_x, new_third_joint_y))
        rectangle1.set_xy((new_rec1_x, new_rec1_y))
        rectangle1.set_angle(local_angle_degrees_second_joint)
        rectangle2.set_xy((new_rec2_x, new_rec2_y))
        rectangle2.set_angle(local_angle_degrees_third_joint)
        angle_degrees_second_joint = local_angle_degrees_second_joint
        angle_degrees_third_joint = local_angle_degrees_third_joint

        plt.draw()
        return True



    return False
    # -----------------------------------------------------------------------------------------------------------






# Function to handle key presses for the second joint
def on_key_second_joint(event):
    global angle_degrees_second_joint
    if event.key == 'a':
        #angle_degrees_second_joint += 10  # Rotate the second joint counterclockwise by 10 degrees
        update_joint_positions(10, 0)
    elif event.key == 'd':
        #angle_degrees_second_joint -= 10  # Rotate the second joint clockwise by 10 degrees
        update_joint_positions(-10, 0)
    #angle_degrees_second_joint %= 360  # Ensure angle stays within 0 to 360 degrees
    #update_joint_positions()


# Function to handle key presses for the third joint
def on_key_third_joint(event):
    global angle_degrees_third_joint
    if event.key == 'left':
        #angle_degrees_third_joint += 10  # Rotate the third joint counterclockwise by 10 degrees
        update_joint_positions(0, 10)
    elif event.key == 'right':
        #angle_degrees_third_joint -= 10  # Rotate the third joint clockwise by 10 degrees
        update_joint_positions(0, -10)
    #angle_degrees_third_joint %= 360  # Ensure angle stays within 0 to 360 degrees
    #update_joint_positions()

print_scene(polygons,ax)

# Connect the key press events to their respective joints
fig.canvas.mpl_connect('key_press_event', on_key_second_joint)
fig.canvas.mpl_connect('key_press_event', on_key_third_joint)

# Show the plot
if(FLAG_START ==0):
    while(1):
        a= update_joint_positions(random.uniform(0, 360), random.uniform(0, 360))
        print(a)
        if(a==True):
            break


FLAG_START == 1
plt.show()
