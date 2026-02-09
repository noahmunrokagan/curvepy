import numpy as np
import segyio
import h5py
import os

class SeismicLoader:
    def __init__(self, filepath):
        """
        Initialize the loader with the path to a .sgy or .segy file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Seismic file not found: {filepath}")
        self.filepath = filepath

    def load_2d_slice(self, inline=None, crossline=None):
        if inline is None and crossline is None:
            raise ValueError("Specify either inline or crossline.")

        with segyio.open(self.filepath, "r", ignore_geometry=True) as f:
            # Define byte locations
            il_byte = segyio.TraceField.INLINE_3D
            xl_byte = segyio.TraceField.CROSSLINE_3D
            
            # 1. Fast-grab all header values for the whole file as a numpy array
            target_byte = il_byte if inline is not None else xl_byte
            target_val = inline if inline is not None else crossline
            
            # This is the "magic" line for unstructured files:
            all_headers = f.attributes(target_byte)[:] 
            
            # 2. Find the indices of the traces we want
            indices = np.where(all_headers == target_val)[0]
            
            if len(indices) == 0:
                # Let's be helpful and show the range if they pick a wrong number
                actual_min, actual_max = np.min(all_headers), np.max(all_headers)
                raise ValueError(f"Value {target_val} not found. Range is {actual_min} to {actual_max}")

            # 3. Pull only those traces
            # f.trace.raw allows us to pull multiple indices at once efficiently
            slice_data = f.trace.raw[indices[0] : indices[-1] + 1]
            
            # 4. Cleanup and Format
            data = slice_data.T.astype(np.float32)
            np.nan_to_num(data, copy=False, nan=0.0)
            
            return data

    def save_coefficients(self, coeffs, filename, threshold=1e-5):
        """
        v1.2 High-Efficiency Sparse Saver.
        Consolidates wedges into scales to reduce HDF5 overhead.
        """
        print(f"Saving super-sparse coefficients to {filename}...")
        
        with h5py.File(filename, 'w') as hf:
            hf.attrs['n_scales'] = len(coeffs)
            hf.attrs['storage_type'] = 'super_sparse_v1.2'

            for s_idx, scale_data in enumerate(coeffs):
                s_grp = hf.create_group(f'scale_{s_idx}')
                
                all_values = []
                all_coords = []
                wedge_info = [] # Store (shape, length) to split them back later

                for w_idx, wedge in enumerate(scale_data):
                    # Find significant points
                    idx = np.where(np.abs(wedge) > threshold)
                    val = wedge[idx].astype(np.complex64)
                    
                    # Store data
                    all_values.append(val)
                    # Use uint16 to save 50% on coordinate storage
                    all_coords.append(np.array(idx, dtype=np.uint16))
                    wedge_info.append([wedge.shape[0], wedge.shape[1], len(val)])

                # Save the scale as a single block (much more efficient for GZIP)
                if all_values:
                    s_grp.create_dataset('values', data=np.concatenate(all_values), 
                                         compression="gzip", compression_opts=9)
                    s_grp.create_dataset('coords', data=np.concatenate(all_coords, axis=1), 
                                         compression="gzip", compression_opts=9)
                    s_grp.create_dataset('metadata', data=np.array(wedge_info, dtype=np.uint32))

    def load_coefficients(self, filename):
        """Loads from the super-sparse v1.2 consolidated format."""
        coeffs = []
        with h5py.File(filename, 'r') as hf:
            n_scales = hf.attrs['n_scales']
            for s in range(n_scales):
                s_grp = hf[f'scale_{s}']
                metadata = s_grp['metadata'][()]
                vals = s_grp['values'][()]
                coords = s_grp['coords'][()]
                
                scale_list = []
                cursor = 0
                for i in range(len(metadata)):
                    rows, cols, length = metadata[i]
                    wedge = np.zeros((rows, cols), dtype=np.complex128)
                    
                    # Extract slice of the big scale-block
                    v_part = vals[cursor : cursor + length]
                    c_part = coords[:, cursor : cursor + length]
                    
                    wedge[c_part[0], c_part[1]] = v_part
                    scale_list.append(wedge)
                    cursor += length
                coeffs.append(scale_list)
        return coeffs
    
    def save_quantized_coefficients(self, coeffs, filename, threshold=1e-5, scale_factor=None):
        """
        v1.3 Quantized Saver. 
        Converts complex floats to Int16 integers to beat the redundancy trap.
        
        ARGS:
            scale_factor (float): If None, auto-calculated from max amplitude.
        """
        print(f"Quantizing and saving to {filename}...")
        
        # 1. Determine Global Scale Factor (if not provided)
        # We need to map the max amplitude to ~32,000 (safe int16 range)
        if scale_factor is None:
            max_val = 0
            for scale in coeffs:
                for wedge in scale:
                    m = np.max(np.abs(wedge))
                    if m > max_val: max_val = m
            # Leave some headroom
            scale_factor = 32000.0 / (max_val + 1e-9)

        with h5py.File(filename, 'w') as hf:
            hf.attrs['n_scales'] = len(coeffs)
            hf.attrs['storage_type'] = 'quantized_v1'
            hf.attrs['scale_factor'] = scale_factor
            
            # Save the max value just in case we need it for statistics
            hf.attrs['max_amplitude'] = max_val 

            for s_idx, scale_data in enumerate(coeffs):
                s_grp = hf.create_group(f'scale_{s_idx}')
                
                # We will flatten the whole scale into big arrays
                all_real = []
                all_imag = []
                all_coords = []
                wedge_info = [] 

                for w_idx, wedge in enumerate(scale_data):
                    # 1. Threshold
                    idx = np.where(np.abs(wedge) > threshold)
                    
                    if len(idx[0]) == 0:
                        wedge_info.append([wedge.shape[0], wedge.shape[1], 0])
                        continue

                    val = wedge[idx]
                    
                    # 2. Quantize (Float -> Int16)
                    # We separate real and imaginary parts to compress better
                    q_real = (val.real * scale_factor).astype(np.int16)
                    q_imag = (val.imag * scale_factor).astype(np.int16)
                    
                    all_real.append(q_real)
                    all_imag.append(q_imag)
                    
                    # 3. Coordinate encoding (uint16 is enough for < 65k dim)
                    all_coords.append(np.array(idx, dtype=np.uint16))
                    wedge_info.append([wedge.shape[0], wedge.shape[1], len(val)])

                # Save optimized datasets
                if all_real:
                    # Int16 compresses VERY well with GZIP
                    s_grp.create_dataset('q_real', data=np.concatenate(all_real), 
                                         compression="gzip", compression_opts=9)
                    s_grp.create_dataset('q_imag', data=np.concatenate(all_imag), 
                                         compression="gzip", compression_opts=9)
                    s_grp.create_dataset('coords', data=np.concatenate(all_coords, axis=1), 
                                         compression="gzip", compression_opts=9)
                    
                s_grp.create_dataset('metadata', data=np.array(wedge_info, dtype=np.uint32))
        
        print(f"Saved with quantization factor: {scale_factor:.2f}")

    def load_quantized_coefficients(self, filename):
        """Reconstructs float coefficients from the quantized Int16 data."""
        coeffs = []
        with h5py.File(filename, 'r') as hf:
            n_scales = hf.attrs['n_scales']
            scale_factor = hf.attrs['scale_factor']
            
            for s in range(n_scales):
                s_grp = hf[f'scale_{s}']
                metadata = s_grp['metadata'][()]
                
                # Handle empty scales (rare but possible)
                if 'q_real' in s_grp:
                    q_real = s_grp['q_real'][()]
                    q_imag = s_grp['q_imag'][()]
                    coords = s_grp['coords'][()]
                    has_data = True
                else:
                    has_data = False

                scale_list = []
                cursor = 0
                
                for i in range(len(metadata)):
                    rows, cols, length = metadata[i]
                    wedge = np.zeros((rows, cols), dtype=np.complex128)
                    
                    if has_data and length > 0:
                        # Extract chunks
                        r_part = q_real[cursor : cursor + length]
                        i_part = q_imag[cursor : cursor + length]
                        c_part = coords[:, cursor : cursor + length]
                        
                        # De-Quantize: Int16 -> Float
                        # (value / scale_factor)
                        reconstructed_val = (r_part + 1j * i_part) / scale_factor
                        
                        wedge[c_part[0], c_part[1]] = reconstructed_val
                        cursor += length
                    
                    scale_list.append(wedge)
                coeffs.append(scale_list)
        return coeffs
    

    def apply_threshold(self, coeffs, threshold_sigma):
        """
        Zeroes out coefficients smaller than threshold_sigma.
        Returns the denoised coeffs and the sparsity percentage.
        """
        denoised_coeffs = []
        total_elements = 0
        kept_elements = 0

        for scale in coeffs:
            new_scale = []
            for wedge in scale:
                # Calculate mask of "important" data
                mask = np.abs(wedge) > threshold_sigma
                
                # Keep only important data, zero the rest
                new_wedge = wedge * mask
                new_scale.append(new_wedge)
                
                # Stats for sparsity
                total_elements += wedge.size
                kept_elements += np.count_nonzero(mask)
                
            denoised_coeffs.append(new_scale)

        sparsity = 100 * (1 - kept_elements / total_elements)
        return denoised_coeffs, sparsity
    
    