import math
import sys
from random import uniform, randint
import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.spatial import ConvexHull



#Normalizes a 2D vector
def normalize(vector):
    length = np.linalg.norm(vector)
    if length == 0:
        return vector
    return vector / length

#Projects a polygon onto an axis and return the minimum and maximum values
def polygon_min_max(polygon, axis):
    min_value = float('inf')
    max_value = -float('inf')
    for vertex in polygon:
        projection = np.dot(vertex, axis)
        min_value = min(min_value, np.min(projection))
        max_value = max(max_value, np.max(projection))
    return min_value, max_value

#Checks if the given axis is a separating axis between two polygons
def is_separating_axis(polygon1, polygon2, axis):
    min1, max1 = polygon_min_max(polygon1, axis)
    min2, max2 = polygon_min_max(polygon2, axis)
    return not (max1 >= min2 and max2 >= min1)

#Checks if two convex polygons are colliding using the Separating Axis Theorem
def check_polygons(vertices1, vertices2):
    polygon1 = np.array(vertices1)
    polygon2 = np.array(vertices2)

    for edge in range(len(polygon1)):
        next_edge = (edge + 1) % len(polygon1)
        edge_vector = polygon1[next_edge] - polygon1[edge]
        axis = np.array([-edge_vector[1], edge_vector[0]])  # Perpendicular to the edge
        if is_separating_axis(polygon1, polygon2, normalize(axis)):
            return False

    for edge in range(len(polygon2)):
        next_edge = (edge + 1) % len(polygon2)
        edge_vector = polygon2[next_edge] - polygon2[edge]
        axis = np.array([-edge_vector[1], edge_vector[0]])  # Perpendicular to the edge
        if is_separating_axis(polygon1, polygon2, normalize(axis)):
            return False

    return True

def collision_detection_full(numpy_verticies):
    polygon_no = numpy_verticies.shape[0]
    array_collision = [0] * polygon_no

    for i in range(polygon_no - 1):
        for j in range(i + 1, polygon_no):
            # if array_collision[j] == 1:
            #     continue
            if check_polygons(numpy_verticies[i], numpy_verticies[j]):
                array_collision[j] = 1
                array_collision[i] = 1
    return array_collision

def print_scene(polygons, ax):
    array_collision = collision_detection_full(polygons)
    # Create a polygon patch using the generated vertices
    for i, vertices in enumerate(polygons):
        polygon = plt.Polygon(vertices, closed=True, edgecolor=('lime' if array_collision[i] else 'teal'),
                              facecolor=('teal' if array_collision[i] else 'none'), alpha=0.5)
        ax.add_patch(polygon)

if __name__ == "__main__":
    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect('equal')
    # load in
    print("For TA copy this file name collision_checking_polygons.npy, i was using input to check different files")
    file = input("file name: ")
    polygons = np.load(file, allow_pickle = True)

    print_scene(polygons,ax)



    # Add the polygon patch to the axis

    # Set axis limits
    ax.set_xlim(0, 2.00)
    ax.set_ylim(0,  2.00)

    # Display the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    a = 1

