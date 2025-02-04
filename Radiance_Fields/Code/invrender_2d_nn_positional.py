# Inverse renderer of 2D images using pixel-wise neural regression with positional encoding
#
# See install_notes for dependencies
# 
# Feb 2025 - Created for / presented at RVSS 2025 by Don Dansereau

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pynput import keyboard

# Tunables
learning_rate = 1e-3
display_interval = 1
batch_size = 1024

num_frequencies = 10  # Adjust based on the desired resolution
layers = [2 + 2*2 * num_frequencies, 128, 128, 3] # Input (u, v, positional encoding), 2 hidden layers, output (R, G, B)
key = random.PRNGKey(42)

# Load target image
target_image_path = 'Images/sunset_128.jpg'  # Update with your image path
target_image = imageio.imread(target_image_path) / 255.0  # Normalize to [0, 1]
target_image = jnp.array(target_image)  # Convert to JAX array

# Image dimensions
height, width, channels = target_image.shape

# Generate a grid of pixel coordinates
coords = jnp.array([(u, v) for v in range(height) for u in range(width)])  # Shape (H*W, 2)
coords = (coords - jnp.array([width / 2, height / 2])) / jnp.array(
    [width / 2, height / 2]
)  # Normalize to [-1, 1]

def positional_encoding(coords, num_frequencies):
    frequencies = jnp.array([2**i for i in range(num_frequencies)])
    sin_encodings = jnp.sin(jnp.pi * coords[..., None] * frequencies).reshape(coords.shape[0], -1)
    cos_encodings = jnp.cos(jnp.pi * coords[..., None] * frequencies).reshape(coords.shape[0], -1)
    return jnp.concatenate([coords, sin_encodings, cos_encodings], axis=-1)

coords_encoded = positional_encoding(coords, num_frequencies)

# Target image reshaped to match coordinate format
target_pixels = target_image.reshape(-1, channels)  # Shape (H*W, 3)

# Define a simple MLP
def initialize_mlp_params(key, layers):
    params = []
    for i in range(len(layers) - 1):
        key, subkey = random.split(key)
        w = random.normal(subkey, (layers[i], layers[i + 1])) * jnp.sqrt(2.0 / layers[i])
        b = jnp.zeros(layers[i + 1])
        print(f"Layer {i + 1}: w shape: {w.shape}, b shape: {b.shape}")  # Debugging print        
        params.append((w, b))
    return params

def mlp_forward(params, x):
    for i, (w, b) in enumerate(params[:-1]):
        #x = jnp.tanh(jnp.dot(x, w) + b)
        x = jax.nn.relu(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jax.nn.sigmoid(jnp.dot(x, w) + b)  # Output values in [0, 1]

# Loss function
@jit
def loss_function(params, coords_encoded, target_pixels):
    predictions = mlp_forward(params, coords_encoded)
    return jnp.mean((predictions - target_pixels) ** 2)

# Training loop
@jit
def train_step(params, opt_state, coords_encoded, target_pixels):
    grads = grad(loss_function)(params, coords_encoded, target_pixels)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Hyperparameters
params = initialize_mlp_params(key, layers)

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Training
num_batches = coords.shape[0] // batch_size

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
while not stop_training:
    epoch += 1

    perm = random.permutation(key, coords_encoded.shape[0])
    coords_shuffled = coords_encoded[perm]
    target_shuffled = target_pixels[perm]

    for batch in range(num_batches):
        batch_coords = coords_shuffled[batch * batch_size : (batch + 1) * batch_size]
        batch_targets = target_shuffled[batch * batch_size : (batch + 1) * batch_size]
        params, opt_state = train_step(params, opt_state, batch_coords, batch_targets)

    # Calculate and log loss
    if epoch % display_interval == 0:
        loss_val = loss_function(params, coords_encoded, target_pixels)
        print(f"Epoch {epoch}, Loss: {loss_val:.6f}")
        generated_image = mlp_forward(params, coords_encoded).reshape(height, width, channels)
        plt.clf()
        plt.imshow(generated_image)
        plt.draw()
        plt.pause(0.00001)

listener.stop()
