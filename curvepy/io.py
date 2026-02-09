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

    def save_coefficients(self, coeffs, filename):
        """
        Saves Curvelet coefficients to an HDF5 file.
        
        Curvelet coeffs are a 'List of Lists of Arrays' (Irregular structure).
        We map this to HDF5 groups:
            /Scale_0
                /Wedge_0
            /Scale_1
                /Wedge_0
                /Wedge_1
                ...
        """
        print(f"Saving coefficients to {filename}...")
        
        with h5py.File(filename, 'w') as hf:
            # Metadata: Store how many scales we have
            hf.attrs['n_scales'] = len(coeffs)

            for scale_idx, scale_data in enumerate(coeffs):
                # Create a Group for this scale (e.g., "scale_0")
                scale_group = hf.create_group(f'scale_{scale_idx}')
                
                # Store how many wedges are in this scale
                scale_group.attrs['n_wedges'] = len(scale_data)

                for wedge_idx, wedge_data in enumerate(scale_data):
                    # Save the actual numpy array as a Dataset
                    scale_group.create_dataset(
                        name=f'wedge_{wedge_idx}', 
                        data=wedge_data,
                        compression="gzip" # Optional: saves disk space
                    )
        print("Save complete.")

    def load_coefficients(self, filename):
        """
        Loads coefficients back from HDF5 into the list-of-lists format.
        """
        coeffs = []
        with h5py.File(filename, 'r') as hf:
            n_scales = hf.attrs['n_scales']
            
            for s in range(n_scales):
                scale_list = []
                scale_group = hf[f'scale_{s}']
                n_wedges = scale_group.attrs['n_wedges']
                
                for w in range(n_wedges):
                    data = scale_group[f'wedge_{w}'][()] # [()] reads the dataset into memory
                    scale_list.append(data)
                
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