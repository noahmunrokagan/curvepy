import numpy as np
cimport numpy as np
from cython cimport floating

# 1. Define Fused Type
ctypedef fused numerical_type:
    np.float64_t
    np.complex128_t

# 2. Wrap Wedge (Safe Signature)
# We remove 'int' from arguments in 'def' to prevent C-compilation errors
# regarding VLA initialization. We cast them to 'cdef int' immediately inside.
def wrap_wedge_fast(numerical_type[:, :] full_grid, 
                   L1, L2,          # <--- Removed 'int' type here
                   cy, cx,          # <--- Removed 'int' type here
                   img_rows, img_cols): # <--- Removed 'int' type here
    
    # Cast to C integers for speed inside the function
    cdef int c_L1 = L1
    cdef int c_L2 = L2
    cdef int c_cy = cy
    cdef int c_cx = cx
    cdef int c_img_rows = img_rows
    cdef int c_img_cols = img_cols
    
    # Allocate output
    cdef int nrows = c_L1
    cdef int ncols = c_L2
    
    cdef np.ndarray result_arr
    if numerical_type is np.float64_t:
        result_arr = np.zeros((nrows, ncols), dtype=np.float64)
    else:
        result_arr = np.zeros((nrows, ncols), dtype=np.complex128)
        
    cdef numerical_type[:, :] result = result_arr
    
    cdef int r, c, src_r, src_c
    cdef int half_nrows = nrows // 2
    cdef int half_ncols = ncols // 2
    
    # The Loop
    for r in range(nrows):
        for c in range(ncols):
            
            # Row Wrapping (Safe Double Modulo)
            src_r = (c_cy + (r - half_nrows)) % c_img_rows
            if src_r < 0:
                src_r += c_img_rows
            
            # Col Wrapping (Safe Double Modulo)
            src_c = (c_cx + (c - half_ncols)) % c_img_cols
            if src_c < 0:
                src_c += c_img_cols
            
            result[r, c] = full_grid[src_r, src_c]
            
    return result_arr

# 3. Unwrap Wedge (Safe Signature)
def unwrap_wedge_fast(numerical_type[:, :] wedge_data, 
                     numerical_type[:, :] target_grid,
                     cy, cx,          # <--- Removed 'int'
                     img_rows, img_cols): # <--- Removed 'int'
    
    # Cast to C integers
    cdef int c_cy = cy
    cdef int c_cx = cx
    cdef int c_img_rows = img_rows
    cdef int c_img_cols = img_cols

    cdef int nrows = wedge_data.shape[0]
    cdef int ncols = wedge_data.shape[1]
    
    cdef int r, c, dst_r, dst_c
    cdef int half_nrows = nrows // 2
    cdef int half_ncols = ncols // 2
    
    for r in range(nrows):
        for c in range(ncols):
            
            # Row Wrapping
            dst_r = (c_cy + (r - half_nrows)) % c_img_rows
            if dst_r < 0:
                dst_r += c_img_rows
            
            # Col Wrapping
            dst_c = (c_cx + (c - half_ncols)) % c_img_cols
            if dst_c < 0:
                dst_c += c_img_cols
            
            # Accumulate
            target_grid[dst_r, dst_c] = target_grid[dst_r, dst_c] + wedge_data[r, c]