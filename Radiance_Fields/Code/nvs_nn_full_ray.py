# Novel view synthesis in flatland using ray-wise neural regression with positional encoding
# For rays in x, y, theta
#
# See install_notes for dependencies
# 
# Feb 2025 - Created for / presented at RVSS 2025 by Don Dansereau

import svg_reader as svg
import renderer2d as renderer
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
import imageio.v2 as imageio
from pynput import keyboard
from model_saving import save_model, load_model
import os

# Tunables
model_filename = os.path.splitext("Trained/" + os.path.basename(__file__))[0] + ".model.pkl"
reset_model = True
test_only = False

filename = "Scenes/scene_simple.svg"

# filename = "Scenes/scene_simple_test.svg"
# reset_model = False
# test_only = True

scene_center = jnp.array([80, 127])
scene_scale = jnp.linalg.norm(scene_center)
rays_per_image = 64
learning_rate = 1e-3
num_frequencies = 10  # For positional encoding
display_interval = 1
epochs = 100
batch_size = 4096

layers = [3, 128, 128, 3]  # Input (2 pos + 1 angle), 2 hidden layers, output (R, G, B)
layers_positional = [3 + 3 * 2 * num_frequencies] + layers[1:] # Adjust input layer size for positional encoding
key = random.PRNGKey(42)

# Positional encoding function
@jit
def positional_encoding(x):
    encodings = [x]
    for i in range(num_frequencies):
        for fn in [jnp.sin, jnp.cos]:
            encodings.append(fn((2.0 ** i) * x))
    return jnp.concatenate(encodings, axis=-1)

# Define a simple MLP
def initialize_mlp_params(key, layers):
    params = []
    for i in range(len(layers) - 1):
        key, subkey = random.split(key)
        w = random.normal(subkey, (layers[i], layers[i + 1])) * jnp.sqrt(2.0 / layers[i])
        b = jnp.zeros(layers[i + 1])
        params.append((w, b))
    return params

# Forward pass with positional encoding
@jit
def forward_pass(params, x):
    x = positional_encoding(x)
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = jax.nn.relu(x)
    w, b = params[-1]
    # x = jnp.dot(x, w) + b  # linear output
    x = jax.nn.sigmoid(jnp.dot(x, w) + b)  # Output values in [0, 1]
    return x

# Loss function
@jit
def loss_function(params, coords, target_pixels):
    predictions = forward_pass(params, coords)
    return jnp.mean((predictions - target_pixels) ** 2)

# Training loop
@jit
def train_step(params, opt_state, coords, target_pixels):
    grads = grad(loss_function)(params, coords, target_pixels)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

if __name__ == "__main__":
    svg_content = svg.read_svg_file(filename)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    target_rendered_images, all_rays = renderer.render_views(fig, ax1, svg_content, rays_per_image)
    ax1.imshow(target_rendered_images, aspect='auto', interpolation='none')
    plt.draw()
    plt.pause(0.00001)

    # Get rendered image into convenient format
    target_rendered_images = jnp.array(target_rendered_images / 255.0)  # Convert to JAX array, normalised
    HEIGHT, WIDTH, CHANNELS = target_rendered_images.shape
    target_pixels = target_rendered_images.reshape(-1, CHANNELS)  # Shape (H*W, 3)

    # Reorganize the all_rays array
    coords = []
    for rays in all_rays:
        for ray in rays:
            coords.append(ray.flatten())
    coords = jnp.array(coords)

    # Interpret the third and fourth coordinate as a direction vector and replace them with a single angle
    directions = coords[:, 2:4]
    angles = jnp.arctan2(directions[:, 1], directions[:, 0])
    # angles = (angles / jnp.pi) # Normalize angles to the range -1 to 1
    # angles = (angles - jnp.min(angles)) / (jnp.max(angles) - jnp.min(angles)) * 2 - 1
    coords = coords.at[:, 2].set(angles)
    coords = coords[:, :3]  # Remove the fourth column

    # Center cameras around origin
    coords = coords.at[:, :2].add(-scene_center)

    # Normalize the ray position to be in the range -1 to 1
    scaling_factor = max(scene_center)
    coords = coords.at[:, :2].divide(scaling_factor)

    # print("Coordinates:")
    # np.set_printoptions(threshold=np.inf)
    # print(np.array(coords))

    # Training
    total_num_rays = sum(len(rays) for rays in all_rays)
    batch_size = min(batch_size, total_num_rays)
    num_batches = total_num_rays // batch_size
    print(f"Total number of rays: {total_num_rays}, batch size: {batch_size}, total batches: {num_batches}")

    # Init parameters
    if not reset_model and os.path.exists(model_filename):
        params = load_model(model_filename)
    else:
        params = initialize_mlp_params(key, layers_positional)  # Adjust input layer size for positional encoding
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    epoch = 0

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

    print("Press 'q' to stop training...")
    while not stop_training and not test_only:
        epoch += 1

        perm = random.permutation(key, coords.shape[0])
        coords_shuffled = coords[perm]
        target_shuffled = target_pixels[perm]

        for batch in range(num_batches):
            batch_coords = coords_shuffled[batch * batch_size : (batch + 1) * batch_size]
            batch_targets = target_shuffled[batch * batch_size : (batch + 1) * batch_size]
            params, opt_state = train_step(params, opt_state, batch_coords, batch_targets)

        # Calculate and log loss
        if epoch % display_interval == 0:
            loss_val = loss_function(params, coords, target_pixels)
            print(f"Epoch {epoch}, Loss: {loss_val:.6f}")
            generated_image = forward_pass(params, coords).reshape(HEIGHT, WIDTH, CHANNELS)
            ax2.clear()
            ax2.imshow(generated_image, aspect='auto', interpolation='none')
            plt.draw()
            plt.pause(0.00001)

    listener.stop()

    if not test_only:
        save_model(params, model_filename)

    # Generate final image
    print("Generating final image")
    generated_image = forward_pass(params, coords).reshape(HEIGHT, WIDTH, CHANNELS)
    ax2.imshow(generated_image, aspect='auto', interpolation='none')
    plt.draw()
    plt.pause(0.00001)

    if test_only:
        input("Press Enter to continue...")

    plt.close('all')
