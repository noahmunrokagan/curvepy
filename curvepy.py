import numpy as np
import matplotlib.pyplot as plt

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

    # # INPUTS: same as above
    # # OUTPUT: 2D Boolean Array (NxN)

    # # LOGIC:
    # 1. Initialize a grid `mask` of zeros (size NxN).
    mask = np.zeros((N,N))

    # 2. Get Outer Square Coordinates:
    #    # CALL get_corona_indices for `scale_idx`
    #    outer_rows, outer_cols = get_corona_indices(N, scale_idx, ...)
    outer_rows, outer_cols = get_corona_indices(N, scale_idx, total_scales)
    # 3. fill outer square:
    #    Set `mask[outer_rows, outer_cols]` = 1

    mask[outer_rows[0]:outer_rows[1], outer_cols[0]:outer_cols[1]] = 1
    # 4. Handle the "Hole" (if not the coarsest scale):
    #    IF scale_idx > 0:
    #        # Get dimensions of the previous (smaller) scale
    #        inner_rows, inner_cols = get_corona_indices(N, scale_idx - 1, ...)

    #        # Carve out the center
    #        Set `mask[inner_rows, inner_cols]` = 0
    if scale_idx > 0:
        inner_rows, inner_cols = get_corona_indices(N, scale_idx - 1, total_scales)

        mask[inner_rows[0]:inner_rows[1], inner_cols[0]:inner_cols[1]] = 0

    return mask

if __name__ == "__main__":
    N = 512
    total_scales = 8
    scale_idx = 5
    mask = get_corona_mask(N, scale_idx, total_scales)
    plt.imshow(mask, cmap="Greys", vmin=0, vmax=1)
    plt.show()




