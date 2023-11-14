import math
import sys
from random import uniform, randint
import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.spatial import ConvexHull




def generate_random_polygon_vertices(num_vertices_range, scene_size, radius_range):
    vertices = []
    num_vertices = randint(num_vertices_range[0], num_vertices_range[1])
    radius = uniform(radius_range[0], radius_range[1])
    center = (uniform(radius, scene_size[0] - radius), uniform(radius, scene_size[1] - radius))

    for _ in range(num_vertices):
        angle = uniform(0, 2 * math.pi)

        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))

    vertices.sort(key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]))

    return vertices





if __name__ == "__main__":
    # Input
    file = input("New file name, end it with .npy: ")
    num_polygons = int(input("Enter how many polygons: "))
    radius_min = float(input("Radius minimum: "))
    radius_max = float(input("Radius max: "))
    vert_min = int(input("Vertex minimum: "))
    vert_max = int(input("Vertex max: "))

    # Create a figure and axis
    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect('equal')
    vertices_list = [generate_random_polygon_vertices((vert_min, vert_max), (2.00, 2.00), (radius_min, radius_max)) for
                     _ in range(num_polygons)]
    numpy_vertices = np.array(vertices_list, dtype=object)

    numpy.save(file, numpy_vertices, allow_pickle=True, fix_imports=True)

    # Create a polygon patch using the generated vertices
    for i, vertices in enumerate(vertices_list):
        polygon = plt.Polygon(vertices, closed=True, edgecolor='teal', facecolor='none')
        ax.add_patch(polygon)

    # Set axis limits
    ax.set_xlim(0, 2.00)
    ax.set_ylim(0, 2.00)

    # Display the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
