import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation

# Constants
L = 0.2  # wheelbase
dt = 0.1  # time step

SCENE_X = 2.0
SCENE_Y = 2.0

def update_configuration(q, v, phi):
    theta = q[2]
    q_dot = np.array([v * math.cos(theta), v * math.sin(theta), (v / L) * math.tan(phi)])
    q_new = q + q_dot * dt
    return q_new

def plot_grid(ax, q):
    ax.clear()
    ax.set_xlim([0, SCENE_X])
    ax.set_ylim([0, SCENE_Y])

    x, y, theta = q
    car = Rectangle((x - 0.1, y - 0.05), 0.2, 0.1, angle=math.degrees(theta),fill=False, color='blue')
    ax.add_patch(car)

    plt.grid(True)

def key_press(event):
    global v, phi
    if event.key == 'up':
        v = min(v + 0.1, 0.5)
    elif event.key == 'down':
        v = max(v - 0.1, -0.5)
    elif event.key == 'left':
        phi = max(phi - 0.1, -math.pi / 4)
    elif event.key == 'right':
        phi = min(phi + 0.1, math.pi / 4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a planar car")
    parser.add_argument("--start", nargs=3, type=float, default=[1.0, 1.0, 0.0], help="Initial configuration (x, y, theta)")
    args = parser.parse_args()

    q = np.array(args.start)
    v, phi = 0.0, 0.0

    fig, ax = plt.subplots(dpi=100)
    ax.set_aspect('equal')
    ax.set_xlim(0, SCENE_X)
    ax.set_ylim(0, SCENE_Y)

    plot_grid(ax, q)
    fig.canvas.mpl_connect('key_press_event', key_press)

    def update(frame):
        global q, v, phi
        q = update_configuration(q, v, phi)
        plot_grid(ax, q)
        print(f"Time: {frame * dt:.2f}s - Configuration: {q}")

    ani = animation.FuncAnimation(fig, update, frames=int(5 / dt), interval=dt * 1000)

    plt.show()
