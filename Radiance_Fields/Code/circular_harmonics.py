# Simple circular harmonics
#
# See install_notes for dependencies
# 
# Feb 2025 - Created for / presented at RVSS 2025 by Don Dansereau

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


def circular_harmonic_sum_rgb(coefficients, theta, max_harmonic):
    indices = jnp.arange(-max_harmonic, max_harmonic + 1)
    cos_terms = jnp.cos(indices * theta)
    sin_terms = jnp.sin(indices * theta)
    
    def harmonic_sum(coefficients):
        A_n = jnp.array([coeff[0] for coeff in coefficients])
        B_n = jnp.array([coeff[1] for coeff in coefficients])
        return jnp.sum(A_n * cos_terms + B_n * sin_terms)
    
    num_harmonics = max_harmonic * 2 + 1
    coefficients_r = coefficients[:num_harmonics]
    coefficients_g = coefficients[num_harmonics:2*num_harmonics]
    coefficients_b = coefficients[2*num_harmonics:3*num_harmonics]
    
    r = harmonic_sum(coefficients_r)
    g = harmonic_sum(coefficients_g)
    b = harmonic_sum(coefficients_b)
    
    return jnp.array([r, g, b])


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    max_harmonic = 2
    num_harmonics = max_harmonic*2 + 1

    coeffs_rgb = [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(num_harmonics * 3)]

    coeffs_rgb = np.random.uniform(-1, 1, (num_harmonics * 3, 2))
    theta_values = np.linspace(0, 2 * np.pi, 1000)
    harmonic_values_rgb = [circular_harmonic_sum_rgb(coeffs_rgb.tolist(), theta, max_harmonic) for theta in theta_values]

    r_values = [val[0] for val in harmonic_values_rgb]
    g_values = [val[1] for val in harmonic_values_rgb]
    b_values = [val[2] for val in harmonic_values_rgb]

    plt.figure()
    plt.plot(theta_values, r_values, 'r', label='Red Channel')
    plt.plot(theta_values, g_values, 'g', label='Green Channel')
    plt.plot(theta_values, b_values, 'b', label='Blue Channel')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Harmonic Sum')
    plt.title('Circular Harmonic Sum RGB over 360 Degrees')
    plt.legend()
    plt.grid(True)

    plt.show()