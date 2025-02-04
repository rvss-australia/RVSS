# Novel view synthesis in flatland using point-based primitives: Gaussian blobs with splatting
# Each blob has position, size, density, and colour c( theta ) as a sum of circular harmonics
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
from jax.numpy import atan2
import optax
import imageio.v2 as imageio
from pynput import keyboard
from model_saving import save_model, load_model
import os
from circular_harmonics import circular_harmonic_sum_rgb

# Tunables
model_filename = os.path.splitext("Trained/" + os.path.basename(__file__))[0] + ".model.pkl"
reset_model = True
test_only = False

filename = "Scenes/scene_simple.svg"

# filename = "Scenes/scene_simple_test.svg"
# reset_model = False
# test_only = True

# filename = "Scenes/scene_refl.svg"

colour_map_samps = 40
scene_center = jnp.array([80, 127])
scene_scale = jnp.linalg.norm(scene_center)

num_blobs = 6        # number of Gaussians to use
max_harmonic = 2
num_harmonics = max_harmonic*2 + 1
mute_harmonics = False

rays_per_image = 16
max_dist = 1.0  # Define the maximum distance for ray integration
min_dist = 0
learning_rate = 1e-2

display_interval = 5
epochs = 100
batch_size = 64

key = random.PRNGKey(42)

# Function to compute visual density at a given 2D coordinate
@jit
def gaussian_2d(coord, center, blobsize, opacity):
    x, y = coord
    cx, cy = center
    return opacity * jnp.exp(-(((x - cx) ** 2) / (2 * blobsize ** 2) + ((y - cy) ** 2) / (2 * blobsize ** 2)))

@jit
def compute_visual_density_and_colour(params, coord, angle):
    accumulated_density = 0.0
    accumulated_color = jnp.zeros(3)
    for blob in params:
        center = blob[:2]
        blobsize = blob[2]
        opacity = blob[3]
        coeffs_rgb = blob[4:].reshape((3*num_harmonics, 2))

        if mute_harmonics:
            angle = 0

        color = circular_harmonic_sum_rgb(coeffs_rgb, angle, max_harmonic)
        density = gaussian_2d(coord, center, blobsize, opacity)

        accumulated_density += density
        accumulated_color += color * density

    accumulated_density = jnp.clip(accumulated_density, 0, 1)
    accumulated_color = jnp.clip(accumulated_color, 0, 1)
    return accumulated_density, accumulated_color

@jit
def compute_visual_density_and_colour_single_blob(blob, coord, angle):
    center = blob[:2]
    blobsize = blob[2]
    opacity = blob[3]
    coeffs_rgb = blob[4:].reshape((3*num_harmonics, 2))

    if mute_harmonics:
        angle = 0

    color = circular_harmonic_sum_rgb(coeffs_rgb, angle, max_harmonic)
    density = gaussian_2d(coord, center, blobsize, opacity)
    color = jnp.clip(color, 0, 1)
    density = jnp.clip(density, 0, 1)

    return density, color

@jit
def integrate_ray(params, position, direction):
    accumulated_color = jnp.zeros(3)  # Start with no light contribution
    t_values = []
    
    for blob in params:
        center = blob[:2]
        
        d = direction
        p = position

        cur_t = jnp.dot(center - p, d) / jnp.dot(d, d)
        t_values.append(cur_t)
    
    # Sort t_values with the farthest blob first and keep track of the order
    t_values = jnp.array(t_values)
    sorted_indices = jnp.argsort(t_values)[::-1]
    sorted_blobs = params[sorted_indices]

    for t, blob in zip(t_values[sorted_indices], sorted_blobs):
        current_position = position + t * direction
        density, color = compute_visual_density_and_colour_single_blob(blob, current_position, atan2(direction[1], direction[0]))
        # if t > min_dist and t < max_dist:
        accumulated_color = color*density + accumulated_color*(1 - density)

    accumulated_color = jnp.clip(accumulated_color, 0, 1)
    return accumulated_color

@jit
def integrate_ray_debug(params, position, direction):
    colour_map = jnp.zeros((colour_map_samps, colour_map_samps, 3)) 
    accumulated_color = jnp.zeros(3)  # Start with no light contribution
    t_values = []
    
    for blob in params:
        center = blob[:2]

        # Calculate t that is along the line and closest to the blob
        # This disregards anisotropy, todo: fix!
        d = direction
        p = position        
        cur_t = jnp.dot(center - p, d) / jnp.dot(d, d)
        t_values.append(cur_t)
    
    # Sort t_values with the farthest blob first and keep track of the order
    t_values = jnp.array(t_values)
    sorted_indices = jnp.argsort(t_values)[::-1]
    sorted_blobs = params[sorted_indices]

    for t, blob in zip(t_values[sorted_indices], sorted_blobs):
        current_position = position + t * direction
        density, color = compute_visual_density_and_colour_single_blob(blob, current_position, atan2(direction[1], direction[0]))
        
        # if t > min_dist and t < max_dist:
        accumulated_color = color*density + accumulated_color*(1 - density)
    
        # Map current_position to the colour_map grid
        grid_x = jnp.clip(jnp.floor((current_position[0] + 1) * (colour_map_samps / 2)).astype(int), 0, colour_map_samps - 1)
        grid_y = jnp.clip(jnp.floor((current_position[1] + 1) * (colour_map_samps / 2)).astype(int), 0, colour_map_samps - 1)
        
        out_color = color * density
        colour_map = colour_map.at[grid_y, grid_x].set(out_color)

    colour_map = jnp.clip(colour_map, 0, 1)
    return colour_map


@jit
def build_density_map(density_params):
    # Generate the grid of coordinates
    x_vals = jnp.linspace(-1, 1, colour_map_samps)
    y_vals = jnp.linspace(-1, 1, colour_map_samps)
    X, Y = jnp.meshgrid(x_vals, y_vals)
    coords = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # Shape: (nsamps^2, 2)

    # Vectorized computation of density values
    compute_density_vectorized = vmap(lambda coord: compute_visual_density_and_colour(density_params, coord, 0)[0])
    density_map_flat = compute_density_vectorized(coords)  # Shape: (nsamps^2,)

    # Reshape to the original grid
    density_map = density_map_flat.reshape((colour_map_samps, colour_map_samps))

    density_map = jnp.clip(density_map, 0, 1)
    return density_map

@jit
def display_colour_map(params, x):
    position = x[:, :2]
    direction = jnp.stack([jnp.cos(x[:, 2]), jnp.sin(x[:, 2])], axis=-1)

    colour_map = jax.vmap(lambda pos, dir: integrate_ray_debug(params, pos, dir))(position, direction)
    total_colour_map = jnp.max(colour_map, axis=0)
    return total_colour_map

# Forward pass
@jit
def forward_pass(params, x):
    # print("Performing forward pass...")
    position = x[:, :2]
    direction = jnp.stack([jnp.cos(x[:, 2]), jnp.sin(x[:, 2])], axis=-1)

    colors = jax.vmap(lambda pos, dir: integrate_ray(params, pos, dir))(position, direction)
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


# Initialize random blob parameters
def init_blobs(NBLOBS):
    blobs = []
    for _ in range(NBLOBS):
        center = np.random.uniform(-0.7, 0.7), np.random.uniform(-0.7, 0.7)
        blobsize = np.random.uniform(0.1, 0.3)
        opacity = np.random.uniform(0.1, 0.3)
        coeffs_rgb = np.random.uniform(-1, 1, (num_harmonics * 3, 2))
        coeffs_rgb = coeffs_rgb.flatten()
        blobs.append(jnp.array([*center, blobsize, opacity, *coeffs_rgb]))
    return jnp.stack(blobs)

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
        params = init_blobs(num_blobs)
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
            plt.draw()
            plt.pause(0.00001)

            # Generate and display 2D density map
            density_map = build_density_map(params)
            ax4.imshow(density_map, cmap='viridis', aspect='auto', interpolation='none')
            plt.draw()
            plt.pause(0.00001)

            colour_map = display_colour_map(params, coords)
            ax3.imshow(colour_map, aspect='auto', interpolation='none')
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
        density_map = build_density_map(params)
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
