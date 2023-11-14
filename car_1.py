import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 0.2  # wheelbase
dt = 0.1  # time step

SCENE_X = 5
SCENE_Y = 5
def update_configuration(q, v, phi):
    theta = q[2]
    q_dot = np.array([v * math.cos(theta), v * math.sin(theta), (v / L) * math.tan(phi)])
    q_new = q + q_dot * dt
    return q_new

def plot_grid(q, ax):
    ax.clear()
    ax.set_xlim([0, SCENE_X])
    ax.set_ylim([0, SCENE_Y])

    x, y, theta = q
    car = plt.Rectangle((x - 0.1, y - 0.05), 0.2, 0.1, angle=math.degrees(theta),fill=False, color='blue')
    ax.add_patch(car)


    plt.pause(dt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a planar car")
    parser.add_argument("--control", nargs=2, type=float, help="Control input (velocity and steering angle)")
    parser.add_argument("--start", nargs=3, type=float, help="Initial configuration (x, y, theta)")
    args = parser.parse_args()

    v, phi = args.control
    q = np.array(args.start)

    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect('equal')
    ax.set_xlim(0, SCENE_X)
    ax.set_ylim(0, SCENE_Y)

    for _ in range(int(5 / dt)):
        q = update_configuration(q, v, phi)
        plot_grid(q, ax)
        print(f"Time: {_ * dt:.2f}s - Configuration: {q}")

    plt.show()