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
from collision_checking import print_scene, check_polygons
import time
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

def get_circle(x, y):
    num_sides = 16
    angles = np.linspace(0, 2 * np.pi, num_sides + 1)[:num_sides]
    return [(x + JOINT_RADIUS * np.cos(angle), y + JOINT_RADIUS * np.sin(angle)) for angle in angles]


def check_joint(arm_vert, polygons):
    polygon_no = polygons.shape[0]
    for i in range(polygon_no):
        if check_polygons(arm_vert, polygons[i]):
            return True
    return False


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

    def __init__(self, add_two, add_three):
        main_joint_coordinates = np.array([START_POINT_X, START_POINT_Y])
        self.main_joint = patches.Circle(main_joint_coordinates, joint_radius, fill=False, edgecolor='black')

        second_joint_coordinates = np.array([main_joint_coordinates[0] + second_joint_radius,
                                             main_joint_coordinates[1]])

        self.second_joint = patches.Circle(second_joint_coordinates, joint_radius, fill=False, edgecolor='black')

        self.rectangle1 = patches.Rectangle(np.array([START_POINT_X + JOINT_RADIUS, START_POINT_Y - JOINT_RADIUS]),
                                            LEN_J1 - (2 * JOINT_RADIUS), JOINT_RADIUS * 2, fill=False,
                                            edgecolor='black')

        self.rec1_dist = calculate_distance(main_joint_coordinates[0], main_joint_coordinates[1],
                                            self.rectangle1.get_x(),
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

        self.rec2_dist = calculate_distance(second_joint_coordinates[0], second_joint_coordinates[1],
                                            self.rectangle2.get_x(),
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
        local_angle_degrees_second_joint = add_two
        local_angle_degrees_third_joint = add_three
        # local_angle_degrees_second_joint %= 360
        # local_angle_degrees_third_joint %= 360

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

    def check_colision(self, obstacles):
        j1 = get_circle(self.main_joint.center[0],self.main_joint.center[1])
        j2 = get_circle(self.second_joint.center[0],self.second_joint.center[1])
        j3 = get_circle(self.third_joint.center[0], self.third_joint.center[1])
        if not check_joint(j1, obstacles) and not check_joint(j2, obstacles) and not check_joint(j3,
                                                                                                 obstacles) and not check_joint(
                self.rec1_ver, obstacles) and not check_joint(self.rec2_ver, obstacles):
            return True
        return False


def interpolate(start, goal, resolution):
    step = (goal - start) / resolution
    for i in range(resolution + 1):
        yield start + i * step





class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


def rrt(start, goal, max_iterations, polygons, step_size=0.3):
    nodes = [Node(start[0], start[1])]

    for _ in range(max_iterations):
        rand_point = (uniform(-3, 3), uniform(-3, 3))
        nearest_node = nodes[0]
        min_distance = math.sqrt((rand_point[0] - nearest_node.x) ** 2 + (rand_point[1] - nearest_node.y) ** 2)

        for node in nodes:
            distance = math.sqrt((rand_point[0] - node.x) ** 2 + (rand_point[1] - node.y) ** 2)
            if distance < min_distance:
                nearest_node = node
                min_distance = distance

        new_node = extend(nearest_node, rand_point, step_size, polygons)
        if new_node is not None:  # Check if extend succeeded
            new_node.parent = nearest_node  # Set the parent of the new node
            new_arm = arm(new_node.x, new_node.y)
            if new_arm.check_colision(polygons):
                nodes.append(new_node)

                if math.sqrt((new_node.x - goal[0]) ** 2 + (new_node.y - goal[1]) ** 2) < 0.2:
                    path = construct_path(new_node)
                    return path, nodes

    return None, nodes



def extend(node, target, step_size, obstacles):
    distance = math.sqrt((target[0] - node.x) ** 2 + (target[1] - node.y) ** 2)
    delta_x = target[0] - node.x
    delta_y = target[1] - node.y

    # Normalize the direction vector
    scale = step_size / distance
    normalized_delta_x = delta_x * scale
    normalized_delta_y = delta_y * scale

    # Check for collisions along the path
    num_steps = int(distance / step_size)
    for i in range(num_steps):
        x = node.x + i * normalized_delta_x
        y = node.y + i * normalized_delta_y
        arm_at_point = arm(x, y)
        if not arm_at_point.check_colision(obstacles):
            continue
        else:
            return None  # Stop extension if collision occurs

    # If no collision, create and return the new node
    x = node.x + normalized_delta_x
    y = node.y + normalized_delta_y
    return Node(x, y)


def construct_path(node):
    path = [(node.x, node.y)]
    while node.parent:
        node = node.parent
        path.insert(0, (node.x, node.y))
    return path

def sort_nodes(nodes, start, end):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    nodes.sort(key=lambda n: n.x, reverse=True if start>end else False)
    return nodes


def interpolate(start, goal, resolution):
    step_x = (goal[0] - start[0]) / resolution
    step_y = (goal[1] - start[1]) / resolution

    for i in range(resolution + 1):
        yield (start[0] + i * step_x, start[1] + i * step_y)

def animate_arm(nodes, the_arm):
    def update(frame):
        the_arm.update_joint_positions(frame[0], frame[1])

    frames = []
    for i in range(len(nodes) - 1):
        start_config = (nodes[i].x, nodes[i].y)
        goal_config = (nodes[i + 1].x, nodes[i + 1].y)
        frames.extend(interpolate(start_config, goal_config, resolution=10))  # Adjust resolution as needed

    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.show()

#------------------------------------------------------
def animate_rrt_tree(start_config, goal_config, nodes, the_arm, path_edges=None):
    def update_rrt_tree(frame):
        for i in range(frame + 1):
            current_node = nodes[i]
            while current_node.parent is not None:
                if path_edges and (current_node, current_node.parent) in path_edges:
                    plt.plot([current_node.x, current_node.parent.x], [current_node.y, current_node.parent.y], 'r-',
                             linewidth=2.0)
                else:
                    plt.plot([current_node.x, current_node.parent.x], [current_node.y, current_node.parent.y], 'g-',
                             linewidth=0.5)
                current_node = current_node.parent

        # Plot nodes as green circles
        for node in nodes:
            plt.plot(node.x, node.y, 'go', markersize=3)

        # Plot goal as a red circle
        plt.plot(goal_config[0], goal_config[1], 'ro', markersize=5)

        # Set plot limits
        plt.xlim(-4, 4)
        plt.ylim(-4,4)
        plt.gca().set_aspect('equal', adjustable='box')

    ani = animation.FuncAnimation(fig, update_rrt_tree, frames=len(nodes), repeat=False)
    plt.show()
#------------------------------------------------------
def animate_rrt_tree(start_config, goal_config, nodes, the_arm, path_edges=None):
    fig, ax = plt.subplots(dpi=100)  # Create a new figure and axis
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')

    def update_rrt_tree(frame):
        for i in range(frame + 1):
            current_node = nodes[i]
            while current_node.parent is not None:
                if path_edges and (current_node, current_node.parent) in path_edges:
                    plt.plot([current_node.x, current_node.parent.x], [current_node.y, current_node.parent.y], 'r-',
                             linewidth=2.0)
                else:
                    plt.plot([current_node.x, current_node.parent.x], [current_node.y, current_node.parent.y], 'g-',
                             linewidth=0.5)
                current_node = current_node.parent

        # Plot nodes as green circles
        for node in nodes:
            plt.plot(node.x, node.y, 'go', markersize=3)

        # Plot goal as a red circle
        plt.plot(goal_config[0], goal_config[1], 'ro', markersize=5)

        # Set plot limits
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.gca().set_aspect('equal', adjustable='box')

    ani = animation.FuncAnimation(fig, update_rrt_tree, frames=len(nodes), repeat=False)
    plt.show()


#------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolation for Robot Arm Configurations")
    parser.add_argument("--start", type=float, nargs=2, help="Start robot configuration (theta1, theta2)")
    parser.add_argument("--goal", type=float, nargs=2, help="Goal robot configuration (theta1, theta2)")
    parser.add_argument("--map", type=str, nargs=1, help="map")
    args = parser.parse_args()
    polygons = np.load(args.map[0], allow_pickle=True)
    start_config = np.array(args.start)
    goal_config = np.array(args.goal)

    ax.set_aspect('equal')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    plt.gca().set_aspect('equal', adjustable='box')

    max_iterations = 1000
    path, nodes = rrt(args.start, args.goal, max_iterations, polygons)

    nodes = [node for node in nodes if min(args.start[0], args.goal[0]) < node.x < max(args.start[0], args.goal[0])]
    nodes = sort_nodes(nodes, args.start[0], args.goal[0])

    new_arm = arm(0, 0)
    new_arm.set_color("red")
    new_arm.do_patch()

    print_scene(polygons, ax)

    # Create and show the arm animation
    arm_animation = animate_arm(nodes, new_arm)

    animate_rrt_tree(start_config, goal_config, nodes, new_arm)






    #----------------------------------------------------
