import numpy as np

def generate_random_configurations(num_configurations):
    # Define the range of joint angles for each joint in radians
    min_angle = 0
    max_angle = 2 * np.pi

    # Generate random joint angles for each configuration
    configurations = np.random.uniform(min_angle, max_angle, size=(num_configurations, 2))

    return configurations

if __name__ == "__main__":
    num_configurations = int(input("Enter the number of configurations: "))
    filename = input("Enter the filename to save configurations (end it with .npy): ")

    # Generate random configurations
    configurations = generate_random_configurations(num_configurations)

    # Save configurations to a file
    np.save(filename, configurations)

    print(f"{num_configurations} random configurations saved to {filename}.")
