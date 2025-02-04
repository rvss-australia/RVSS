# Inverse renderer of 2D images using point-based primitives: Gaussian blobs
#
# See install_notes for dependencies
# 
# Feb 2025 - Created for / presented at RVSS 2025 by Don Dansereau

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from jax import grad, jit
import optax
from pynput import keyboard

# constants
target_image_path = 'Images/sunset_128.jpg'

nblobs = 32           # number of Gaussians to use
output_period = 1     # how often to update display

np.random.seed(42)

# Function to generate a Gaussian blob
# Anisotropic but axis-aligned only, i.e. no rotation
def gaussian_blob(center, size_x, size_y, color, x, y):
    dx = (x - center[0]) / size_x
    dy = (y - center[1]) / size_y
    opacity = jnp.exp(-(dx**2 + dy**2) / 2)[:, :, None] 
    return opacity, color

# Function to generate the image from blobs
def generate_image(blob_params, height, width):
    x_grid, y_grid = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    image = jnp.zeros((height, width, 3))
    for blob in blob_params:
        cur_opacity, cur_color = gaussian_blob(
            center=blob[0:2],
            size_x=blob[2],
            size_y=blob[3],
            color=blob[5:8],
            x=x_grid,
            y=y_grid,
        )
        image = jnp.where(cur_opacity > 0.1, cur_color*cur_opacity, image) # TODO: try this, no blend
        # image += cur_color * cur_opacity # simple blend
    return jnp.clip(image, 0.0, 1.0)

# Loss function: Mean Squared Error
@jit
def loss_function(blob_params, target_image):
    generated_image = generate_image(blob_params, target_image.shape[0], target_image.shape[1])
    return jnp.mean((generated_image - target_image) ** 2)

# Initialize random blob parameters
def init_blobs(NBLOBS, height, width):
    blobs = []
    for _ in range(NBLOBS):
        center = np.random.uniform(0, width), np.random.uniform(0, height)
        size_x = np.random.uniform(5, 10)
        size_y = np.random.uniform(5, 10)
        opacity = np.random.uniform(0.2, 0.5)
        color = np.random.uniform(-1, 1, size=3)
        blobs.append(jnp.array([*center, size_x, size_y, opacity, *color]))
    return jnp.stack(blobs)

# Load the target image
target_image = imageio.imread(target_image_path) / 255.0  # Normalize to [0, 1]
target_image = jnp.array(target_image)  # Convert to JAX array
height, width = target_image.shape[:2]  # Image dimensions
   
# Initialize blobs and optimizer
blob_params = init_blobs(nblobs, height, width)
optimizer = optax.adam(learning_rate=1e-1)
opt_state = optimizer.init(blob_params)

# Training loop
stop_training = False
def on_press(key):
    global stop_training
    try:
        if key.char == 'q':  # Stop training if 'q' is pressed
            stop_training = True
    except AttributeError:
        pass # Handle special keys if necessary

# Start the listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

step = 0
while not stop_training:
    step = step + 1
    
    grads = grad(loss_function)(blob_params, target_image)
    updates, opt_state = optimizer.update(grads, opt_state)
    blob_params = optax.apply_updates(blob_params, updates)

    if step % output_period == 0:
        loss_val = loss_function(blob_params, target_image)
        print(f"\tStep {step}, Loss: {loss_val:.6f}")
        generated_image = generate_image(blob_params, height, width)
        plt.clf()
        plt.imshow(np.array(generated_image))
        plt.draw()
        plt.pause(0.00001)

listener.stop()
