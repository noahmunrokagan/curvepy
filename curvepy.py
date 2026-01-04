import numpy as np
import matplotlib.pyplot as plt

BASE_WEDGES = 8

def is_power_of_two(n: int) -> bool:
    return n > 0 and n.bit_count() == 1

def get_corona_indices(N: int, scale_idx: int, total_scales: int):
    """
    Calculates slice indices for a specific frequency scale

    
    # INPUTS:
    # N             -> Integer, dimension of the image, must be power of 2 (e.g., 512)
    # scale_idx     -> Integer, index of concentric shells, 0 is coarsest (center), 
    #                  total_scales-1 is finest (border)
    # total_scales  -> Integer, total number of concentric shells

    # OUTPUTS:
    # row_indices   -> Tuple (start, end) for the rows
    # col_indices   -> Tuple (start, end) for the columns
    """
    if not is_power_of_two(N):
        raise ValueError("Input image dimension must be a power of 2")
    center = N // 2

    base_radius = N // (2 ** total_scales) # center to outside of shell

    if scale_idx == 0:
        radius = base_radius
    else:
        radius = base_radius * (2 ** scale_idx)

    # Compute boundaries
    row_start = center - radius
    row_end = center + radius
    col_start = center - radius
    col_end = center + radius

    # Ensure no invalid indices
    if (row_start < 0 or col_start < 0) or (row_end > N or col_end > N):
        raise ValueError("An unknown error occured, radius of value: {radius} incompatible with image with dimensions: {N}")
    
    
    return (row_start, row_end), (col_start, col_end)


def get_corona_mask(N: int, scale_idx: int, total_scales: int) -> np.ndarray:
    """
    Creates a binary mask (1 inside the shell, 0 outside) for visualization.

    # INPUTS:
    # N             -> Integer, dimension of the image, must be power of 2 (e.g., 512)
    # scale_idx     -> Integer, index of concentric shells, 0 is coarsest (center), 
    #                  total_scales-1 is finest (border)
    # total_scales  -> Integer, total number of concentric shells

    # OUTPUTS:
    # mask          -> np.ndarray, 2D array visualizing the shell computed
    """
    
    # Initialize mask of zeros
    mask = np.zeros((N,N))

    # Get outer square coordinates
    outer_rows, outer_cols = get_corona_indices(N, scale_idx, total_scales)

    # Fill outer square
    mask[outer_rows[0]:outer_rows[1], outer_cols[0]:outer_cols[1]] = 1
   
    # Handle hole if present
    if scale_idx > 0:
        inner_rows, inner_cols = get_corona_indices(N, scale_idx - 1, total_scales)

        mask[inner_rows[0]:inner_rows[1], inner_cols[0]:inner_cols[1]] = 0

    return mask


def get_wedge_slope_ranges(scale_idx: int) -> np.ndarray:
    """
    Returns a list of slope boundaries for the 'East' Quadrant at a given scale. Other quadrant boundaries can be found from this array.
    """
    
    # 1. Handle the Coarse Scale Exception
    if scale_idx == 0:
        return None

    # Number of wedges with parabolic scaling
    steps = int(np.floor(scale_idx / 2))
    num_wedges = BASE_WEDGES * (2 ** steps)

    # Slope ranges
    min_slope = -1.0
    max_slope = +1.0
    
    # Generate boundaries
    slope_boundaries = np.linspace(min_slope, max_slope, num_wedges + 1)
    
    return slope_boundaries

# if __name__ == "__main__":
#     N = 512
#     total_scales = 8
#     scale_idx = 5
#     mask = get_corona_mask(N, scale_idx, total_scales)
#     slope_ranges = get_wedge_slope_ranges(scale_idx)
#     plt.imshow(mask, cmap="Greys", vmin=0, vmax=1)
#     plt.show()



# if __name__ == "__main__":
#     import matplotlib.colors as mcolors # Helper for colors

#     N = 512
#     total_scales = 5
#     scale_idx = 2  # Try 2 or 3 to see the wedges clearly
    
#     # 1. Get the container (The Ring)
#     mask = get_corona_mask(N, scale_idx, total_scales)
#     slope_ranges = get_wedge_slope_ranges(scale_idx)
    
#     # 2. Create a Coordinate Grid (Centered)
#     # y is row index (centered), x is col index (centered)
#     y, x = np.mgrid[-N//2:N//2, -N//2:N//2]
    
#     # 3. Calculate Slopes for every pixel
#     # Avoid divide by zero by adding a tiny number
#     slopes = y / (x + 1e-10)
    
#     # 4. Filter for the "East Cone"
#     # The East Cone is where x > 0 and |slope| <= 1
#     east_cone_mask = (x > 0) & (np.abs(slopes) <= 1)
    
#     # 5. Bin the pixels into Wedges
#     # np.digitize checks which "slope_range" bucket each pixel falls into
#     # We assign IDs 1, 2, 3... to the wedges.
#     wedge_ids = np.digitize(slopes, slope_ranges)
    
#     # 6. Combine it all for display
#     # Start with a background (0)
#     final_image = np.zeros((N, N))
    
#     # Paint the Corona Gray (value 0.5) so we see the ring
#     final_image[mask == 1] = 0.5 
    
#     # Paint the Wedges on top (Values 1, 2, 3...)
#     # We only paint pixels that are in the Ring AND in the East Cone
#     active_pixels = (mask == 1) & east_cone_mask
#     final_image[active_pixels] = wedge_ids[active_pixels]

#     # 7. Plotting
#     plt.figure(figsize=(8, 8))
#     # Use a colormap that highlights different numbers (tab20 is good)
#     plt.imshow(final_image, cmap="tab20b", origin='upper')
#     plt.title(f"Scale {scale_idx}: East Quadrant Wedges")
#     plt.colorbar(label="Wedge ID")
#     plt.show()
