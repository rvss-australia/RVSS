# Novel view synthesis in flatland using ray-wise neural regression with positional encoding
# Soft surface light field given by density in x,y and colour in x,y,theta
# Ray march and apply alpha blending
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
from jax import grad, jit, vmap, random
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

# filename = "Scenes/scene_refl.svg"

colour_map_samps = 80
scene_center = jnp.array([80, 127])
scene_scale = jnp.linalg.norm(scene_center)

rays_per_image = 64
max_dist = 1.0  # Define the maximum distance for ray integration
step_resolution = 0.1
min_dist = 0
learning_rate = 1e-3

num_frequencies = 10  # For positional encoding
display_interval = 1
epochs = 100
batch_size = 64

layers_density = [2 + 2 * 2 * num_frequencies, 128, 128, 1]  # Input (2 pos), 2 hidden layers, output (density)
layers_colour = [4 + 4 * 2 * num_frequencies, 128, 128, 3]  # Input (2 pos + 1 angle), 2 hidden layers, output (R, G, B)
key = random.PRNGKey(42)

# Positional encoding function
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

# Function to compute visual density at a given 2D coordinate
@jit
def compute_visual_density(params, coord):
    coord = jnp.array(coord).reshape(1, -1)  # Ensure input is a 2D array with shape (1, 2)
    coord = positional_encoding(coord)  # Apply positional encoding
    for w, b in params[:-1]:
        coord = jnp.dot(coord, w) + b
        coord = jax.nn.relu(coord)
    w, b = params[-1]
    density = jax.nn.sigmoid(jnp.dot(coord, w) + b)  # Output value in [0, 1]
    # print(f"Coord: {coord}, Density: {density}")
    return density[0, 0]  # Return the scalar density value

# Forward pass for a single position and direction
@jit
def compute_colour(params, position, direction):
    coord = jnp.array([position, direction]).reshape(1, -1)  # Ensure input is a 2D array with shape (1, 2)
    coord = positional_encoding(coord)  # Apply positional encoding
    for w, b in params[:-1]:
        coord = jnp.dot(coord, w) + b
        coord = jax.nn.relu(coord)
    w, b = params[-1]
    rgb = jax.nn.sigmoid(jnp.dot(coord, w) + b)  # Output values in [0, 1]
    # print(f"Position: {position}, Direction: {direction}, RGB: {rgb}")
    return rgb[0]  # Return the RGB color

@jit
def integrate_ray(density_params, colour_params, position, direction):
    accumulated_color = jnp.zeros(3)  # Start with no light contribution
    t = max_dist

    while t > min_dist:
        current_position = position + t * direction
        density = compute_visual_density(density_params, current_position)
        color = compute_colour(colour_params, current_position, direction)
        
        accumulated_color = color * density + accumulated_color * (1 - density)

        t -= step_resolution

    return accumulated_color

@jit
def integrate_ray_debug(density_params, colour_params, position, direction):
    colour_map = jnp.zeros((colour_map_samps, colour_map_samps, 3))  # Use static value for shape
    accumulated_color = jnp.zeros(3)  # Start with no light contribution
    t = max_dist

    while t > min_dist:
        current_position = position + t * direction
        density = compute_visual_density(density_params, current_position)
        color = compute_colour(colour_params, current_position, direction)
        
        accumulated_color = color * density + accumulated_color * (1 - density)
    
        # Map current_position to the colour_map grid
        grid_x = jnp.clip(jnp.floor((current_position[0] + 1) * (colour_map_samps / 2)).astype(int), 0, colour_map_samps - 1)
        grid_y = jnp.clip(jnp.floor((current_position[1] + 1) * (colour_map_samps / 2)).astype(int), 0, colour_map_samps - 1)
        
        out_color = color * density
        colour_map = colour_map.at[grid_y, grid_x].set(out_color)

        t -= step_resolution

    return colour_map

@jit
def build_density_map(density_params):
    # Generate the grid of coordinates
    x_vals = jnp.linspace(-1, 1, colour_map_samps)
    y_vals = jnp.linspace(-1, 1, colour_map_samps)
    X, Y = jnp.meshgrid(x_vals, y_vals)
    coords = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # Shape: (nsamps^2, 2)

    # Vectorized computation of density values
    compute_density_vectorized = vmap(lambda coord: compute_visual_density(density_params, coord))
    density_map_flat = compute_density_vectorized(coords)  # Shape: (nsamps^2,)

    # Reshape to the original grid
    density_map = density_map_flat.reshape((colour_map_samps, colour_map_samps))

    return density_map


@jit
def display_colour_map(params, x):
    density_params, colour_params = params
    position = x[:, :2]
    direction = jnp.stack([jnp.cos(x[:, 2]), jnp.sin(x[:, 2])], axis=-1)

    colour_map = jax.vmap(lambda pos, dir: integrate_ray_debug(density_params, colour_params, pos, dir))(position, direction)
    total_colour_map = jnp.max(colour_map, axis=0)
    return total_colour_map


# Forward pass with positional encodingc
@jit
def forward_pass(params, x):
    # print("Performing forward pass...")
    density_params, colour_params = params
    position = x[:, :2]
    direction = jnp.stack([jnp.cos(x[:, 2]), jnp.sin(x[:, 2])], axis=-1)

    colors = jax.vmap(lambda pos, dir: integrate_ray(density_params, colour_params, pos, dir))(position, direction)
    return colors

# Loss function
@jit
def loss_function(params, coords, target_pixels):
    predictions = forward_pass(params, coords)
    mse_loss = jnp.mean((predictions - target_pixels) ** 2)
    return mse_loss

# Training loop
@jit
def train_step(params, opt_state, coords, target_pixels):
    grads = grad(loss_function)(params, coords, target_pixels)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


if __name__ == "__main__":
    svg_content = svg.read_svg_file(filename)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 6))

    target_rendered_images, all_rays = renderer.render_views(fig, ax1, svg_content, rays_per_image)
    ax1.imshow(target_rendered_images, aspect='auto', interpolation='none')
    ax1.set_title("Input Images")
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

    # Training
    total_num_rays = sum(len(rays) for rays in all_rays)
    batch_size = min(batch_size, total_num_rays)
    num_batches = total_num_rays // batch_size
    print(f"Total number of rays: {total_num_rays}, batch size: {batch_size}, total batches: {num_batches}")

    # Init parameters
    if not reset_model and os.path.exists(model_filename):
        params = load_model(model_filename)
    else:
        density_params = initialize_mlp_params(key, layers_density)
        colour_params = initialize_mlp_params(key, layers_colour)
        params = (density_params, colour_params)
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
        perm = random.permutation(key, coords.shape[0])
        coords_shuffled = coords[perm]
        target_shuffled = target_pixels[perm]

        for batch in range(num_batches):
            batch_coords = coords_shuffled[batch * batch_size : (batch + 1) * batch_size]
            batch_targets = target_shuffled[batch * batch_size : (batch + 1) * batch_size]
            params, opt_state = train_step(params, opt_state, batch_coords, batch_targets)

        epoch += 1

        # Calculate and log loss
        if epoch % display_interval == 0:
            loss_val = loss_function(params, coords, target_pixels)
            print(f"Epoch {epoch}, Loss: {loss_val:.6f}")
            
            generated_image = forward_pass(params, coords).reshape(HEIGHT, WIDTH, CHANNELS)
            ax2.clear()
            ax2.imshow(generated_image, aspect='auto', interpolation='none')
            ax2.set_title("Predicted Images")
            plt.draw()
            plt.pause(0.00001)

            # Generate and display 2D density map
            density_params, colour_params = params
            density_map = build_density_map(density_params)
            ax4.imshow(density_map, cmap='viridis', aspect='auto', interpolation='none')
            ax4.set_title("Predicted Density")
            plt.draw()
            plt.pause(0.00001)

            colour_map = display_colour_map(params, coords)
            ax3.imshow(colour_map, aspect='auto', interpolation='none')
            ax3.set_title("Predicted (prominent) Colour")
            plt.draw()
            plt.pause(0.00001)

    listener.stop()

    if not test_only:
        save_model(params, model_filename)

    if test_only:
        generated_image = forward_pass(params, coords).reshape(HEIGHT, WIDTH, CHANNELS)
        ax2.clear()
        ax2.imshow(generated_image, aspect='auto', interpolation='none')
        ax2.set_title("Predicted Images")
        plt.draw()
        plt.pause(0.00001)

        # Generate and display 2D density map
        density_params, colour_params = params
        density_map = build_density_map(density_params)
        ax4.imshow(density_map, cmap='viridis', aspect='auto', interpolation='none')
        ax4.set_title("Predicted Density")
        plt.draw()
        plt.pause(0.00001)

        colour_map = display_colour_map(params, coords)
        ax3.imshow(colour_map, aspect='auto', interpolation='none')
        ax3.set_title("Predicted (prominent) Colour")
        plt.draw()
        plt.pause(0.00001)

        input("Press Enter to continue...")

    plt.close('all')
