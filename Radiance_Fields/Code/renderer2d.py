# Flatland 1D camera rendering from SVG files
# Most of the meat (and limitations) are in the svg_reader
#
# See install_notes for dependencies
# 
# Feb 2025 - Created for / presented at RVSS 2025 by Don Dansereau

import matplotlib.pyplot as plt
import svg_reader as svg
import numpy as np

def generate_rays_for_camera(camera, rays_per_image=33):
    _, center, angle, direction = camera
    half_angle = np.radians(angle / 2)
    rays = []
    for i in range(rays_per_image):
        ray_angle = -half_angle + (2 * half_angle * i / (rays_per_image - 1))
        ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)]) @ np.array([[direction[0], direction[1]], [-direction[1], direction[0]]])
        rays.append((center, ray_direction))
    return rays

def render_views(fig, ax, elements, rays_per_image=100, debug_display_level = 0):
    numCameras = 0
    all_rendered_images = []
    all_rays = []
    for element in elements:
        if element[0] == 'camera':
            numCameras += 1
            rays = generate_rays_for_camera(element, rays_per_image)
            rendered_image = np.zeros((rays_per_image, 3), dtype=np.uint8)
            
            for i, ray in enumerate(rays):
                closest_intersection, intersection_color = svg.find_intersection(elements, ray)

                if debug_display_level > 10:
                    origin, direction = ray
                    if debug_display_level > 20:
                        ax.plot([origin[0], origin[0] + direction[0] * 100], [origin[1], origin[1] + direction[1] * 100], 'r', zorder=1)
                    if closest_intersection is not None:
                        if intersection_color is not None:
                            ax.plot(closest_intersection[0], closest_intersection[1], 'o', color=np.array(intersection_color) / 255, zorder=100)
                        else:
                            ax.plot(closest_intersection[0], closest_intersection[1], 'go', zorder=100)

                rendered_image[i] = intersection_color if intersection_color is not None else (0, 0, 0)
            all_rendered_images.append(rendered_image)
            all_rays.append(rays)

    all_rendered_images = np.array(all_rendered_images)
    all_rays = np.array(all_rays)
  
    return all_rendered_images, all_rays

if __name__ == "__main__":
    filename = "Scenes/flatland_intro.svg"
    svg_content = svg.read_svg_file(filename)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    svg.draw_elements(fig, ax1, svg_content)
    ax1.invert_yaxis()  # Invert the y-axis to flip the drawing vertically
    ax1.autoscale()
    ax1.set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(0.001)

    rendered_images, rays = render_views(fig, ax1, svg_content, rays_per_image = 16, debug_display_level = 19)
    plt.imshow(rendered_images)
    ax2.autoscale()
    plt.show()
