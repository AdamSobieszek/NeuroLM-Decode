
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
from pathlib import Path
import re
from scipy import signal # For bandpass filtering
import sys # For __main__ example for matplotlib check

class EEGPickleDataset(Dataset):
    def __init__(self, filepaths, expected_channels=None, expected_sfreq=None,
                 model_patch_length=25, model_patch_stride=6,
                 map_filename_to_subject_id=True, default_subject_id=0,
                 transform=None,
                 load_classification_label=False,
                 num_classes_for_label=None,
                 apply_multi_band_mixing=True,
                 mix_probabilities=None,
                 filter_order=5,
                 iterate_all_bands_per_file=False, # New flag
                 desired_processing_duration_sec=3.0, # Duration for cropping/padding,
                 fixed_range = None
                ):
        """
        Args:
            # ... (previous args) ...
            apply_multi_band_mixing (bool): If True, enables band-specific processing.
            mix_probabilities (list/tuple, optional): Probabilities for selecting [raw, alpha, beta, gamma]
                                                      if not iterating all bands. Defaults to uniform.
            filter_order (int): Order of the Butterworth filter for bandpass.
            iterate_all_bands_per_file (bool): If True and apply_multi_band_mixing is True,
                                               the dataset length becomes 4x, iterating through
                                               each band for every file. mix_probabilities is ignored.
            desired_processing_duration_sec (float): The target duration in seconds for EEG segments
                                                     after loading (cropping or padding will be applied).
        """
        self.original_filepaths = filepaths
        self.expected_channels = expected_channels
        self.expected_sfreq = expected_sfreq
        self.model_patch_length = model_patch_length
        self.model_patch_stride = model_patch_stride
        self.map_filename_to_subject_id = map_filename_to_subject_id
        self.default_subject_id = default_subject_id
        self.transform = transform
        self.load_classification_label = load_classification_label
        self.num_classes_for_label = num_classes_for_label
        self.desired_processing_duration_sec = desired_processing_duration_sec
        self.fixed_range = fixed_range

        self.apply_multi_band_mixing = apply_multi_band_mixing
        self.filter_order = filter_order
        self.iterate_all_bands_per_file = iterate_all_bands_per_file
        
        self._samples_info = [] # Stores dicts: {'filepath': str, 'forced_band_key': str or None, 'original_index': int}

        if not self.original_filepaths:
            raise ValueError("Filepaths list cannot be empty.")

        if self.apply_multi_band_mixing:
            if self.expected_sfreq is None:
                raise ValueError("expected_sfreq must be provided when apply_multi_band_mixing is True for filtering.")
            self.bands = {
                "raw": (1.0, 50.0), # LBLM paper: "raw EEG waves, which we filtered between 1 to 50Hz"
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 50.0) # LBLM uses up to 50Hz for gamma
            }
            self.band_keys = list(self.bands.keys()) # Consistent order: raw, alpha, beta, gamma
            print(f"Multi-band processing enabled. Defined bands: {self.bands}")

            if self.iterate_all_bands_per_file:
                if mix_probabilities is not None:
                    print("Info: 'mix_probabilities' is ignored when 'iterate_all_bands_per_file' is True.")
                for i, fp in enumerate(self.original_filepaths):
                    self._samples_info.append({'filepath': fp, 'forced_band_key': "raw", 'original_index': i})
                    band = np.random.choice(["alpha", "beta", "gamma"])
                    self._samples_info.append({'filepath': fp, 'forced_band_key': band, 'original_index': i})
                print(f"Mode: Iterating through all {len(self.band_keys)} bands for each of the {len(self.original_filepaths)} files.")
            else: # Random mixing mode
                if mix_probabilities is None:
                    self.mix_probabilities = [1.0/len(self.band_keys)] * len(self.band_keys) # Uniform
                else:
                    if len(mix_probabilities) != len(self.band_keys) or not np.isclose(sum(mix_probabilities), 1.0):
                        raise ValueError("mix_probabilities must sum to 1 and match number of bands.")
                    self.mix_probabilities = mix_probabilities
                print(f"Mode: Randomly selecting a band for each file with probabilities: {dict(zip(self.band_keys, self.mix_probabilities))}")
                for i, fp in enumerate(self.original_filepaths):
                     self._samples_info.append({'filepath': fp, 'forced_band_key': None, 'original_index': i}) # None means random choice later
        else: # apply_multi_band_mixing is False
            if self.iterate_all_bands_per_file:
                print("Warning: 'iterate_all_bands_per_file' is True but 'apply_multi_band_mixing' is False. No band-specific filtering will occur.")
            if mix_probabilities is not None:
                print("Warning: 'mix_probabilities' is provided but 'apply_multi_band_mixing' is False. It will be ignored.")
            print("Multi-band processing is disabled. Data will be used as loaded from files.")
            for i, fp in enumerate(self.original_filepaths):
                self._samples_info.append({'filepath': fp, 'forced_band_key': None, 'original_index': i})

        print(f"EEGPickleDataset initialized with {len(self.original_filepaths)} original files.")
        print(f"Effective dataset size: {len(self._samples_info)} samples.")
        
        if self.expected_sfreq:
            print(f"Dataset assumes data is at {self.expected_sfreq} Hz.")
            if self.apply_multi_band_mixing and self.expected_sfreq < 100: # Nyquist for 50Hz
                 print(f"Warning: expected_sfreq {self.expected_sfreq}Hz is low for filtering up to 50Hz (Nyquist for 50Hz is 100Hz).")


    def __len__(self):
        return len(self._samples_info)

    def _extract_subject_id_from_filename(self, filepath):
        filename = Path(filepath).name
        match = re.search(r'sub-([0-9]+)', filename)
        if match: return int(match.group(1)) # Consider if +1 offset is truly needed
        parts = filename.split('_')
        for part in parts:
            if part.startswith('s') and len(part) > 1 and part[1:].isdigit():
                try: return int(part[1:])
                except ValueError: continue

        files = ['resultsYAK_NR.mat'
,'resultsYAC_NR.mat'
,'resultsYDG_NR.mat'
,'resultsYAG_NR.mat'
,'resultsYDR_NR.mat'
,'resultsYFR_NR.mat'
,'resultsYFS_NR.mat'
,'resultsYHS_NR.mat'
,'resultsYIS_NR.mat'
,'resultsYLS_NR.mat'
,'resultsYMD_NR.mat'
,'resultsYMS_NR.mat'
,'resultsYRH_NR.mat'
,'resultsYRK_NR.mat'
,'resultsYRP_NR.mat'
,'resultsYSD_NR.mat'
,'resultsYSL_NR.mat'
,'resultsYTL_NR.mat'
,'resultsYAC_TSR.mat'
,'resultsYAG_TSR.mat'
,'resultsYAK_TSR.mat'
,'resultsYDG_TSR.mat'
,'resultsYDR_TSR.mat'
,'resultsYFR_TSR.mat'
,'resultsYFS_TSR.mat'
,'resultsYHS_TSR.mat'
,'resultsYIS_TSR.mat'
,'resultsYLS_TSR.mat'
,'resultsYMD_TSR.mat'
,'resultsYMS_TSR.mat'
,'resultsYRH_TSR.mat'
,'resultsYRK_TSR.mat'
,'resultsYRP_TSR.mat'
,'resultsYSD_TSR.mat'
,'resultsYTL_TSR.mat'
,'resultsYSL_TSR.mat'
,'resultsZAB_NR.mat'
,'resultsZDN_NR.mat'
,'resultsZGW_NR.mat'
,'resultsZJM_NR.mat'
,'resultsZJN_NR.mat'
,'resultsZDM_NR.mat'
,'resultsZKB_NR.mat'
,'resultsZKH_NR.mat'
,'resultsZKW_NR.mat'
,'resultsZMG_NR.mat'
,'resultsZJS_NR.mat'
,'resultsZPH_NR.mat'
,'resultsZAB_SR.mat'
,'resultsZDM_SR.mat'
,'resultsZDN_SR.mat'
,'resultsZGW_SR.mat'
,'resultsZJM_SR.mat'
,'resultsZJN_SR.mat'
,'resultsZJS_SR.mat'
,'resultsZKB_SR.mat'
,'resultsZKH_SR.mat'
,'resultsZKW_SR.mat'
,'resultsZMG_SR.mat'
,'resultsZPH_SR.mat']+['20250226T145925',
'20241222T175331',
'20241222T182917',
'20250126T160415',
'20250204T170301']
        for j, f in enumerate(files):
            if f.replace("results", "").replace(".mat", "") in filepath:
                return j+15
        return self.default_subject_id

    def _bandpass_filter(self, data_ch, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        if high >= 1.0: high = 0.999 
        if low <= 0.0: low = 0.001  
        
        if low >= high: 
            # print(f"Warning: Invalid band for filter: low {lowcut}, high {highcut} at fs {fs}. Returning original data for channel.")
            return data_ch.copy() # Return a copy to avoid modifying original if it's passed around

        try:
            if low <= 0.001 and high < 1.0: 
                 b, a = signal.butter(order, high, btype='low', analog=False)
            elif high >= 0.999 and low > 0.0: 
                 b, a = signal.butter(order, low, btype='high', analog=False)
            else: 
                 b, a = signal.butter(order, [low, high], btype='band', analog=False)
            
            y = signal.lfilter(b, a, data_ch)
            return y
        except Exception as e:
            # print(f"Warning: Filtering failed for band {lowcut}-{highcut}Hz (fs={fs}). Error: {e}. Returning original data for channel.")
            return data_ch.copy()


    def __getitem__(self, idx):
        sample_info = self._samples_info[idx]
        target = -1
        filepath = sample_info['filepath']
        forced_band_key = sample_info.get('forced_band_key')

        try:
            with open(filepath, 'rb') as f:
                data_dict = pickle.load(f)
        except Exception as e:
            raise IOError(f"Error loading pickle file {filepath}: {e}")

        eeg_data_np_orig = data_dict.get("X")
        num_timepoints_orig = eeg_data_np_orig.shape[1]

        if self.expected_sfreq:
            target_timepoints_for_model = int(self.desired_processing_duration_sec * self.expected_sfreq)
        else: 
            # Fallback: estimate based on patches if sfreq not known (less ideal)
            num_patches_desired = 100 # Example: aim for 100 patches
            target_timepoints_for_model = self.model_patch_length + (num_patches_desired - 1) * self.model_patch_stride
            if num_timepoints_orig < target_timepoints_for_model:
                 print(f"Warning: Fallback target_timepoints_for_model ({target_timepoints_for_model}) is greater than available timepoints ({num_timepoints_orig}) in {filepath}. Using available timepoints.")
                 target_timepoints_for_model = num_timepoints_orig


        if isinstance(self.fixed_range, tuple):
            lowcut, highcut = self.fixed_range
            eeg_data_np_orig =  eeg_data_np_orig[:, lowcut:highcut]
        else:
            if num_timepoints_orig >= target_timepoints_for_model:

                if self.fixed_range=="set" and "/tol_" in filepath.lower():
                    eeg_data_np_orig = eeg_data_np_orig[:, 100:700]
                elif self.fixed_range=="set" and "/protocol" in filepath.lower():
                    eeg_data_np_orig = eeg_data_np_orig[:, 0:600]
                else:
                    start_idx = np.random.randint(0, num_timepoints_orig - target_timepoints_for_model + 1)
                    eeg_data_np_orig = eeg_data_np_orig[:, start_idx : start_idx + target_timepoints_for_model]


        
        if eeg_data_np_orig is None:
            raise ValueError(f"Pickle file {filepath} does not contain 'X' key or it's None.")

        if isinstance(eeg_data_np_orig, np.ndarray):
            # normalize and scale to 10
            eeg_data_np_orig = torch.from_numpy(eeg_data_np_orig)
            eeg_norm = torch.norm(eeg_data_np_orig, p=2, dim=(0,1), keepdim=True).clamp(min=1e-9)
            eeg_data_np_orig = (eeg_data_np_orig / eeg_norm) * 100
            eeg_data_np_orig = eeg_data_np_orig.numpy().astype(np.float32)

        elif isinstance(eeg_data_np_orig, torch.Tensor):
            eeg_norm = torch.norm(eeg_data_np_orig, p=2, dim=(0,1), keepdim=True).clamp(min=1e-9)
            eeg_data_np_orig = (eeg_data_np_orig / eeg_norm) * 100
            eeg_data_np_orig = eeg_data_np_orig.numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported data type in {filepath}: {type(eeg_data_np_orig)}")

        if eeg_data_np_orig.ndim != 2:
            raise ValueError(f"EEG data in {filepath} has incorrect ndim: {eeg_data_np_orig.ndim}. Expected 2 (channels x timepoints).")

        if eeg_data_np_orig.shape[1] == target_timepoints_for_model:
            eeg_data_np = eeg_data_np_orig
        elif eeg_data_np_orig.shape[1] < target_timepoints_for_model:
            # print(f"Warning: file {filepath} has {num_timepoints_orig} timepoints, less than target {target_timepoints_for_model}. Padding with zeros.")
            padding_needed = target_timepoints_for_model - eeg_data_np_orig.shape[1]
            eeg_data_np = np.pad(eeg_data_np_orig, ((0,0), (0, padding_needed)), 'constant', constant_values=0)
        else:
            raise ValueError(f"EEG data in {filepath} has {num_timepoints_orig} timepoints after processing, greater than target {target_timepoints_for_model}. This should not happen if target_timepoints_for_model is correctly calculated.")

        num_loaded_channels_orig, num_timepoints_orig = eeg_data_np_orig.shape

        if self.expected_channels is not None and num_loaded_channels_orig != self.expected_channels:
            raise ValueError(f"EEG data in {filepath} has {num_loaded_channels_orig} channels, but expected {self.expected_channels}.")
        

        current_num_channels, current_num_timepoints = eeg_data_np.shape

        if current_num_timepoints < self.model_patch_length:
            raise ValueError(f"Processed EEG data in {filepath} (after crop/pad) has {current_num_timepoints} timepoints, "
                             f"less than model_patch_length ({self.model_patch_length}). This should not happen if target_timepoints_for_model is correctly calculated.")

        final_eeg_data_to_use = eeg_data_np
        actual_band_applied = "input_as_is" # For returning in sample dict

        if self.apply_multi_band_mixing:
            chosen_band_key = None
            if self.iterate_all_bands_per_file:
                chosen_band_key = forced_band_key
                if chosen_band_key is None: # Should not happen due to __init__ logic
                    raise RuntimeError(f"Internal error: forced_band_key is None for {filepath} in iterate_all_bands_per_file mode.")
            else: # Random mixing mode
                chosen_band_key = np.random.choice(self.band_keys, p=self.mix_probabilities)
            
            actual_band_applied = chosen_band_key
            lowcut, highcut = self.bands[chosen_band_key]
            filtered_eeg_data = np.zeros_like(eeg_data_np)
            for ch_idx in range(current_num_channels):
                filtered_eeg_data[ch_idx, :] = self._bandpass_filter(
                    eeg_data_np[ch_idx, :], lowcut, highcut, self.expected_sfreq, self.filter_order
                )
            final_eeg_data_to_use = filtered_eeg_data
        
        eeg_tensor = torch.from_numpy(final_eeg_data_to_use.copy()) # Use .copy() to avoid issues if final_eeg_data_to_use is a view

        subject_id = self.default_subject_id
        if self.map_filename_to_subject_id:
            subject_id = self._extract_subject_id_from_filename(filepath)
        subject_id_tensor = torch.tensor(subject_id, dtype=torch.long)


        if "y" in data_dict:
            target = data_dict["y"]+ (9 if not "word" in data_dict else 1)
        sample = {
            'x_raw_eeg': eeg_tensor*10,  # TODO check this scaling
            'subject_id': subject_id_tensor,
            'filepath': filepath, # For debugging
            'band_applied': actual_band_applied, # For debugging/analysis
            'target': target, # For debugging/analysis
        }
        if "file_name" in data_dict:
            sample['file_name'] = data_dict["file_name"]
        else:
            sample['file_name'] = ""

        if self.load_classification_label:
            label = data_dict.get("y")
            if label is None:
                # print(f"Warning: Label 'y' not found in {filepath}. Using default label 0.")
                label_tensor = torch.tensor(0, dtype=torch.long) 
            elif isinstance(label, (int, float, np.number)):
                label_tensor = torch.tensor(label, dtype=torch.long)
            elif isinstance(label, np.ndarray):
                if label.size == 1: label = label.item() # Convert single item array to scalar
                label_tensor = torch.tensor(label, dtype=torch.long)
            elif isinstance(label, torch.Tensor):
                if label.numel() == 1: label = label.item()
                label_tensor = torch.tensor(label, dtype=torch.long)
            else:
                raise TypeError(f"Unsupported label type {type(label)} in {filepath}")
            sample['y'] = label_tensor

        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage (outside the class)
if __name__ == '__main__':
    print("Testing EEGPickleDataset with Multi-band Mixing options...")
    dummy_data_dir = Path("./dummy_eeg_pickles_mixing_v2")
    dummy_data_dir.mkdir(exist_ok=True)
    
    num_dummy_files = 2
    dummy_channels = 3 
    dummy_sfreq = 200   
    dummy_timepoints_per_file = int(2.5 * dummy_sfreq) # 2.5 seconds of data per file
    
    dummy_filepaths = []
    for i in range(num_dummy_files):
        subject_num_in_fname = i + 1 # Make subject IDs 1-based for testing _extract_subject_id
        filename = f"dummy_mix_sub-{subject_num_in_fname:02d}_segment_{i}.pkl"
        filepath = dummy_data_dir / filename
        
        t = np.linspace(0, dummy_timepoints_per_file / dummy_sfreq, dummy_timepoints_per_file, endpoint=False)
        # Mix of 10Hz (alpha), 20Hz (beta), 40Hz (gamma)
        signal_comp = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.3 * np.sin(2 * np.pi * 40 * t)
        dummy_eeg_data = np.random.randn(dummy_channels, dummy_timepoints_per_file).astype(np.float32) * 0.1 
        dummy_eeg_data += signal_comp 
        
        with open(filepath, 'wb') as f:
            pickle.dump({"X": dummy_eeg_data, "y": i % 2}, f) 
        dummy_filepaths.append(str(filepath))

    print(f"Created {len(dummy_filepaths)} dummy pickle files in {dummy_data_dir}")

    dataset_params_base = {
        "filepaths": dummy_filepaths,
        "expected_channels": dummy_channels,
        "expected_sfreq": dummy_sfreq,
        "model_patch_length": 50, # 0.25s @ 200Hz
        "model_patch_stride": 25, # 0.125s @ 200Hz
        "map_filename_to_subject_id": True,
        "load_classification_label": True,
        "desired_processing_duration_sec": 2.0 # Crop/pad to 2s
    }

    print("\n--- Test Case 1: Iterate all bands per file ---")
    params_iterate_all = dataset_params_base.copy()
    params_iterate_all.update({
        "apply_multi_band_mixing": True,
        "iterate_all_bands_per_file": True,
    })
    
    try:
        dataset_iterate = EEGPickleDataset(**params_iterate_all)
        print(f"Dataset length (iterate_all_bands=True): {len(dataset_iterate)}")
        expected_len = num_dummy_files * len(dataset_iterate.band_keys)
        assert len(dataset_iterate) == expected_len, f"Expected length {expected_len}, got {len(dataset_iterate)}"
        
        print("Checking first few samples to verify band iteration and subject ID:")
        num_bands = len(dataset_iterate.band_keys)
        for i in range(min(len(dataset_iterate), num_bands * 2)): # Check up to 2 files' worth of bands
            sample = dataset_iterate[i]
            original_file_idx = i // num_bands
            expected_band = dataset_iterate.band_keys[i % num_bands]
            print(f"Sample {i}: File {Path(sample['filepath']).name}, SubjectID {sample['subject_id'].item()}, Label {sample['y'].item()}, Expected Band '{expected_band}', Actual Band '{sample['band_applied']}', Shape {sample['x_raw_eeg'].shape}")
            assert sample['band_applied'] == expected_band
            assert sample['x_raw_eeg'].shape[1] == int(params_iterate_all["desired_processing_duration_sec"] * dummy_sfreq)
            # Check subject ID corresponds to original file index (assuming sub-XX in filename)
            expected_subj_id_from_filename = original_file_idx + 1 
            assert sample['subject_id'].item() == expected_subj_id_from_filename


    except Exception as e:
        print(f"Error during iterate_all_bands test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test Case 2: Random band mixing per file ---")
    params_random_mix = dataset_params_base.copy()
    params_random_mix.update({
        "apply_multi_band_mixing": True,
        "iterate_all_bands_per_file": False,
        "mix_probabilities": [0.7, 0.1, 0.1, 0.1] # Favor raw
    })
    try:
        dataset_random = EEGPickleDataset(**params_random_mix)
        print(f"Dataset length (random_mix): {len(dataset_random)}")
        assert len(dataset_random) == num_dummy_files

        band_counts = {band: 0 for band in dataset_random.band_keys}
        num_samples_to_check = min(len(dataset_random) * 5, 50) # Check more if dataset small
        print(f"Checking {num_samples_to_check} random samples (with replacement by re-indexing):")
        for i in range(num_samples_to_check):
            sample_idx = i % len(dataset_random)
            # Set seed for this specific sample to ensure random crop is same if we were to compare to original
            # but primarily, this is to test band selection probabilities over many calls
            # np.random.seed(i) # This seed affects cropping AND band choice if random choice is inside __getitem__
            sample = dataset_random[sample_idx] # This call will have its own np.random.choice for band
            # np.random.seed(None) 
            
            print(f"Sample {sample_idx}: File {Path(sample['filepath']).name}, Band '{sample['band_applied']}', Shape {sample['x_raw_eeg'].shape}")
            if sample['band_applied'] in band_counts:
                band_counts[sample['band_applied']] += 1
        print(f"Band distribution over {num_samples_to_check} samples: {band_counts}")
        # Check if raw is favored (roughly)
        # Note: with few samples, this is not a strict statistical test.
        if num_samples_to_check > 20 and band_counts["raw"] < (num_samples_to_check * 0.4): # Expect ~70%
             print(f"Warning: 'raw' band count ({band_counts['raw']}) seems lower than expected given probabilities.")


    except Exception as e:
        print(f"Error during random_mix test: {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Test Case 3: No multi-band mixing (data as is) ---")
    params_no_mix = dataset_params_base.copy()
    params_no_mix.update({
        "apply_multi_band_mixing": False,
    })
    try:
        dataset_no_mix = EEGPickleDataset(**params_no_mix)
        print(f"Dataset length (no_mix): {len(dataset_no_mix)}")
        assert len(dataset_no_mix) == num_dummy_files
        sample0 = dataset_no_mix[0]
        print(f"Sample 0 (no_mix): File {Path(sample0['filepath']).name}, Band '{sample0['band_applied']}', Shape {sample0['x_raw_eeg'].shape}")
        assert sample0['band_applied'] == "input_as_is"

    except Exception as e:
        print(f"Error during no_mix test: {e}")
        import traceback
        traceback.print_exc()


    # Plotting example for one sample from the iterate_all_bands dataset
    if 'matplotlib' in sys.modules and 'matplotlib.pyplot' in sys.modules : # Check if already imported
        import matplotlib.pyplot as plt
    elif 'matplotlib' not in sys.modules :
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
            print("\nMatplotlib not found, skipping plot generation.")

    if plt and len(dummy_filepaths) > 0 :
        print("\nGenerating plot for the first file, 'alpha' band processing...")
        try:
            dataset_plot_test = EEGPickleDataset(**params_iterate_all) # Use iterate_all settings
            
            # To make crop deterministic for plotting comparison
            np.random.seed(42) 
            # Get the sample corresponding to the first file, second band (alpha)
            # band_keys order is typically ["raw", "alpha", "beta", "gamma"]
            alpha_band_idx_in_keys = dataset_plot_test.band_keys.index("alpha")
            sample_alpha_processed = dataset_plot_test[alpha_band_idx_in_keys] 
            np.random.seed(None) # Reset seed

            # Load original data for comparison
            original_file_path = dummy_filepaths[0]
            with open(original_file_path, 'rb') as f:
                original_data_dict = pickle.load(f)
            original_eeg_from_file = original_data_dict['X'].astype(np.float32)

            # Apply the same cropping to the original data that __getitem__ did
            target_tpts = int(params_iterate_all["desired_processing_duration_sec"] * dummy_sfreq)
            orig_tpts = original_eeg_from_file.shape[1]
            
            original_data_segment_for_plot = None
            if orig_tpts >= target_tpts:
                np.random.seed(42) # Use the same seed as for sample_alpha_processed
                start_idx_plot = np.random.randint(0, orig_tpts - target_tpts + 1)
                original_data_segment_for_plot = original_eeg_from_file[:, start_idx_plot : start_idx_plot + target_tpts]
                np.random.seed(None)
            else: # Padding case
                padding_needed_plot = target_tpts - orig_tpts
                original_data_segment_for_plot = np.pad(original_eeg_from_file, ((0,0), (0, padding_needed_plot)), 'constant', constant_values=0)

            fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            ch_to_plot = 0
            time_axis = np.arange(target_tpts) / dummy_sfreq

            axs[0].plot(time_axis, original_data_segment_for_plot[ch_to_plot, :], label=f"Original Cropped (Ch {ch_to_plot})")
            axs[0].set_title(f"Original Data (Cropped to {params_iterate_all['desired_processing_duration_sec']}s) - File: {Path(original_file_path).name}")
            axs[0].legend()
            axs[0].grid(True, linestyle='--', alpha=0.7)

            axs[1].plot(time_axis, sample_alpha_processed['x_raw_eeg'][ch_to_plot, :].numpy(), label=f"Processed '{sample_alpha_processed['band_applied']}' (Ch {ch_to_plot})")
            axs[1].set_title(f"Data after '{sample_alpha_processed['band_applied']}' filter and processing")
            axs[1].set_xlabel("Time (s)")
            axs[1].legend()
            axs[1].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plot_save_path = dummy_data_dir / "alpha_band_filter_test.png"
            plt.savefig(plot_save_path)
            print(f"Saved plot to {plot_save_path}")
            plt.close(fig)

        except Exception as e:
            print(f"Error during plotting test: {e}")
            import traceback
            traceback.print_exc()

