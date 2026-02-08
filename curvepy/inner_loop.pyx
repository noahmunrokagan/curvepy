# curvepy/inner_loop.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

# We use "fused types" to handle both float (real) and complex numbers efficiently
ctypedef fused numerical_type:
    np.float64_t
    np.complex128_t

def wrap_wedge_fast(numerical_type[:, :] full_grid, 
                   int L1, int L2, 
                   int cy, int cx, 
                   int grid_rows, int grid_cols):
    """
    Direct memory copy for wrapping. 
    Avoids rolling the full m x n array.
    """
    cdef int nrows = L1
    cdef int ncols = L2
    
    # Allocate output array (small wedge)
    cdef np.ndarray result_arr
    if numerical_type is np.float64_t:
        result_arr = np.zeros((nrows, ncols), dtype=np.float64)
    else:
        result_arr = np.zeros((nrows, ncols), dtype=np.complex128)
        
    cdef numerical_type[:, :] result = result_arr
    
    # Calculate offsets
    # The "center" of the output is (nrows//2, ncols//2)
    # The "center" of the input is (cy, cx)
    
    cdef int r, c, src_r, src_c
    cdef int half_nrows = nrows // 2
    cdef int half_ncols = ncols // 2
    
    # We iterate over the SMALL destination grid (The 32x64 wedge)
    # Instead of the BIG source grid (The 512x512 image)
    for r in range(nrows):
        for c in range(ncols):
            
            # 1. Where is this pixel relative to the wedge center?
            # if r=0 (top), rel_y = -half_nrows
            # if r=nrows (bottom), rel_y = +half_nrows
            
            # 2. Where is that in the big grid?
            # We want pixel (cy + rel_y, cx + rel_x)
            # BUT: We must handle the periodic wrapping (The "Torus" topology)
            
            src_r = (cy + (r - half_nrows)) % grid_rows
            src_c = (cx + (c - half_ncols)) % grid_cols
            
            # 3. Direct Copy
            result[r, c] = full_grid[src_r, src_c]
            
    return result_arr

def unwrap_wedge_fast(numerical_type[:, :] wedge_data, 
                     numerical_type[:, :] target_grid,
                     int cy, int cx,
                     int grid_rows, int grid_cols):
    """
    Inverse operation: Pastes small wedge into big grid with wrapping.
    Modifies target_grid in-place (accumulates).
    """
    cdef int nrows = wedge_data.shape[0]
    cdef int ncols = wedge_data.shape[1]
    
    cdef int r, c, dst_r, dst_c
    cdef int half_nrows = nrows // 2
    cdef int half_ncols = ncols // 2
    
    for r in range(nrows):
        for c in range(ncols):
            
            dst_r = (cy + (r - half_nrows)) % grid_rows
            dst_c = (cx + (c - half_ncols)) % grid_cols
            
            # Accumulate (Add to existing grid)
            target_grid[dst_r, dst_c] = target_grid[dst_r, dst_c] + wedge_data[r, c]