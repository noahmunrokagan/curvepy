import numpy as np
import matplotlib.pyplot as plt


def get_corona_indices(N, scale_idx, total_scales):
    """
    Calculates slice indices for a specific frequency scale

    TODO: include ref to img which visualizes this
    
    # INPUTS:
    # N             -> Integer, dimension of the image (e.g., 512)
    # scale_idx     -> Integer, 0 is coarsest (center), total_scales-1 is finest (border)
    # total_scales  -> Integer, total number of concentric shells

    # OUTPUTS:
    # row_indices   -> Tuple (start, end) for the rows
    # col_indices   -> Tuple (start, end) for the columns
    """
    # LOGIC:
    # 1. Define `center` = N / 2
    center = N / 2

    # 2. Define `base_radius`:
    #    # The size of the smallest inner square (coarsest scale).
    #    # Usually N / 2^(total_scales - 1) or a fixed small size like 16.

    base_radius = N / (2 ** total_scales) # rule of thumb derived from

    # 3. Calculate `current_radius`:
    #    # The paper uses dyadic scaling (powers of 2)[cite: 278, 348].
    #    IF scale_idx == 0:
    #        radius = base_radius
    #    ELSE:
    #        # Radius doubles for each subsequent scale
    #        radius = base_radius * (2 ^ scale_idx)
    if scale_idx == 0:
        radius = base_radius
    else:
        radius = base_radius * (2 ** scale_idx)

    # 4. Compute Boundaries:
    #    # Convert radius to array indices centered at N/2
    #    row_start = center - radius
    #    row_end   = center + radius
    #    col_start = center - radius
    #    col_end   = center + radius
    row_start = center - radius
    row_end = center + radius
    col_start = center - radius
    col_end = center + radius

    # 5. Clamp Coordinates:
    #    # Ensure indices don't go below 0 or above N
    #    Ensure starts >= 0
    #    Ensure ends <= N
    if (row_start < 0 or col_start < 0) or (row_end > N or col_end > N):
        raise ValueError
    
    return (row_start, row_end), (col_start, col_end)


def get_corona_mask(N, scale_idx, total_scales):
    # """
    # Creates a binary mask (1 inside the shell, 0 outside) for visualization.
    # """

    # # INPUTS: same as above
    # # OUTPUT: 2D Boolean Array (NxN)

    # # LOGIC:
    # 1. Initialize a grid `mask` of zeros (size NxN).
    mask = np.zeros((N,N))

    # 2. Get Outer Square Coordinates:
    #    # CALL get_corona_indices for `scale_idx`
    #    outer_rows, outer_cols = get_corona_indices(N, scale_idx, ...)
    row_idxs, col_idxs = get_corona_indices(scale_idx, total_scales)
    # 3. Fill Outer Square:
    #    Set `mask[outer_rows, outer_cols]` = 1
    mask[:,:] = 1
    mask[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]] = 0
    # 4. Handle the "Hole" (if not the coarsest scale):
    #    IF scale_idx > 0:
    #        # Get dimensions of the previous (smaller) scale
    #        inner_rows, inner_cols = get_corona_indices(N, scale_idx - 1, ...)

    #        # Carve out the center
    #        Set `mask[inner_rows, inner_cols]` = 0
    if scale_idx > 0:


    # RETURN mask


