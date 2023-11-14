import argparse
import math
import sys
from random import uniform, randint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.animation as animation

# Define the number of interpolation steps (resolution)
SCENE_X = 2.00
SCENE_Y = 2.00

START_POINT_X = 1
START_POINT_Y = 1

JOINT_RADIUS = SCENE_X * 0.025
LEN_J1 = SCENE_X * 0.2
LEN_J2 = SCENE_X * 0.125

joint_radius = JOINT_RADIUS
second_joint_radius = LEN_J1
third_joint_radius = LEN_J2

main_joint_coordinates = np.array([START_POINT_X, START_POINT_Y])
fig, ax = plt.subplots(dpi=100)

num_steps = 100

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

class arm:

  angle_degrees_second_joint = 0
  angle_degrees_third_joint = 0



  def __init__(self,add_two, add_three):
    main_joint_coordinates = np.array([START_POINT_X, START_POINT_Y])
    self.main_joint = patches.Circle(main_joint_coordinates, joint_radius, fill=False, edgecolor='black')


    second_joint_coordinates = np.array([main_joint_coordinates[0] + second_joint_radius,
                                         main_joint_coordinates[1]])

    self.second_joint = patches.Circle(second_joint_coordinates, joint_radius, fill=False, edgecolor='black')


    self.rectangle1 = patches.Rectangle(np.array([START_POINT_X + JOINT_RADIUS, START_POINT_Y - JOINT_RADIUS]),
                                   LEN_J1 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, fill=False, edgecolor='black')


    self.rec1_dist = calculate_distance(main_joint_coordinates[0], main_joint_coordinates[1], self.rectangle1.get_x(),
                                   self.rectangle1.get_y())
    self.rec1_angle = calculate_angle_between_points(
      [START_POINT_X + JOINT_RADIUS, START_POINT_Y - JOINT_RADIUS],
      [START_POINT_X, START_POINT_Y],
      [START_POINT_X + JOINT_RADIUS, START_POINT_Y]
    )

    self.rec1_ver = calculate_rectangle_vertices((START_POINT_X + JOINT_RADIUS, START_POINT_Y - JOINT_RADIUS),
                                            LEN_J1 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, 0)

    third_joint_coordinates = np.array([second_joint_coordinates[0] + third_joint_radius,
                                        second_joint_coordinates[1]])
    self.third_joint = patches.Circle(third_joint_coordinates, joint_radius, fill=False, edgecolor='black')


    # ------------------------------------
    self.rectangle2 = patches.Rectangle(
      np.array([second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1] - JOINT_RADIUS]),
      LEN_J2 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, fill=False, edgecolor='black')


    self.rec2_dist = calculate_distance(second_joint_coordinates[0], second_joint_coordinates[1], self.rectangle2.get_x(),
                                   self.rectangle2.get_y())

    self.rec2_angle = calculate_angle_between_points(
      [second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1] - JOINT_RADIUS],
      [second_joint_coordinates[0], second_joint_coordinates[1]],
      [second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1]]
    )

    self.rec2_ver = calculate_rectangle_vertices(
      (second_joint_coordinates[0] + JOINT_RADIUS, second_joint_coordinates[1] - JOINT_RADIUS),
      LEN_J2 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, 0)

    self.update_joint_positions(add_two, add_three)
    self.joint_config()

  def get_joint1_center(self):
    return self.main_joint.center

  def get_joint2_center(self):
    return self.second_joint.center

  def get_joint3_center(self):
    return self.third_joint.center




  def joint_config(self):
    self.node = np.array([self.angle_degrees_second_joint, self.angle_degrees_third_joint])

  def update_joint_positions(self, add_two, add_three):
    local_angle_degrees_second_joint =  add_two
    local_angle_degrees_third_joint = add_three
    #local_angle_degrees_second_joint %= 360
    #local_angle_degrees_third_joint %= 360

    # Calculate new position for the second joint in a circular motion around the main joint
    angle_radians_second_joint = add_two
    new_second_joint_x = main_joint_coordinates[0] + second_joint_radius * np.cos(angle_radians_second_joint)
    new_second_joint_y = main_joint_coordinates[1] + second_joint_radius * np.sin(angle_radians_second_joint)

    # Calculate new position for the third joint in a circular motion around the second joint
    angle_radians_third_joint = add_three
    new_third_joint_x = new_second_joint_x + third_joint_radius * np.cos(angle_radians_third_joint)
    new_third_joint_y = new_second_joint_y + third_joint_radius * np.sin(angle_radians_third_joint)

    new_rec1_x = main_joint_coordinates[0] + self.rec1_dist * np.cos(angle_radians_second_joint - self.rec1_angle)
    new_rec1_y = main_joint_coordinates[1] + self.rec1_dist * np.sin(angle_radians_second_joint - self.rec1_angle)

    new_rec2_x = new_second_joint_x + self.rec2_dist * np.cos(angle_radians_third_joint - self.rec2_angle)
    new_rec2_y = new_second_joint_y + self.rec2_dist * np.sin(angle_radians_third_joint - self.rec2_angle)

    self.rec1_ver = calculate_rectangle_vertices((new_rec1_x, new_rec1_y),
                                            LEN_J1 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2,
                                            local_angle_degrees_second_joint)
    self.rec2_ver = calculate_rectangle_vertices((new_rec2_x, new_rec2_y),
                                            LEN_J2 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2,
                                            local_angle_degrees_third_joint)

    self.angle_degrees_second_joint = local_angle_degrees_second_joint
    self.angle_degrees_third_joint = local_angle_degrees_third_joint

    self.second_joint.set_center((new_second_joint_x, new_second_joint_y))
    self.third_joint.set_center((new_third_joint_x, new_third_joint_y))
    self.rectangle1.set_xy((new_rec1_x, new_rec1_y))
    self.rectangle1.set_angle(math.degrees(local_angle_degrees_second_joint))
    self.rectangle2.set_xy((new_rec2_x, new_rec2_y))
    self.rectangle2.set_angle(math.degrees(local_angle_degrees_third_joint))

  def do_patch(self):
    ax.add_patch(self.main_joint)
    ax.add_patch(self.second_joint)
    ax.add_patch(self.rectangle1)
    ax.add_patch(self.third_joint)
    ax.add_patch(self.rectangle2)

  def set_color(self, color):
    self.main_joint.set(edgecolor=color)
    self.second_joint.set(edgecolor=color)
    self.rectangle1.set(edgecolor=color)
    self.third_joint.set(edgecolor=color)
    self.rectangle2.set(edgecolor=color)


def interpolate(start, goal, resolution):
    step = (goal - start) / resolution
    for i in range(resolution + 1):
        yield start + i * step


def animate_arm(start_config, goal_config, resolution, the_arm):
    def update(frame):
        the_arm.update_joint_positions(frame[0], frame[1])
        return the_arm.main_joint, the_arm.second_joint, the_arm.rectangle1, the_arm.third_joint, the_arm.rectangle2

    frames = list(interpolate(start_config, goal_config, resolution))

    for frame in frames:
        update(frame)
        plt.pause(0.1)  # Adjust the pause duration as needed
        plt.draw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolation for Robot Arm Configurations")
    parser.add_argument("--start", type=float, nargs=2, help="Start robot configuration (theta1, theta2)")
    parser.add_argument("--goal", type=float, nargs=2, help="Goal robot configuration (theta1, theta2)")
    args = parser.parse_args()

    start_config = np.array(args.start)
    goal_config = np.array(args.goal)

    ax.set_aspect('equal')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    plt.gca().set_aspect('equal', adjustable='box')

    new_arm = arm(0, 0)
    new_arm.set_color("red")
    new_arm.do_patch()

    animate_arm(start_config, goal_config, num_steps, new_arm)
    plt.show()