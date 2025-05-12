# File: eeg_pickle_dataset.py

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
from pathlib import Path
import re # For extracting subject ID from filename

class EEGPickleDataset(Dataset):
    def __init__(self, filepaths, expected_channels=None, expected_sfreq=None,
                 model_patch_length=25, model_patch_stride=6,
                 map_filename_to_subject_id=True, default_subject_id=0,
                 transform=None):
        """
        PyTorch Dataset for loading EEG data from pickle files.

        Args:
            filepaths (list): List of string paths to .pkl files.
            expected_channels (int, optional): If provided, checks if loaded data has this many channels.
                                               Raises error if mismatch.
            expected_sfreq (int, optional): The sampling frequency the model expects the data to be at.
                                            The TUH dataset script resamples to 200Hz.
                                            The LBLM paper uses 250Hz for pre-training.
                                            This dataset does NOT resample; it assumes data is already at target sfreq.
            model_patch_length (int): Patch length used by the LBLM model. Used to check if
                                      timepoints are sufficient for at least one patch.
            model_patch_stride (int): Patch stride used by the LBLM model. (Not directly used here,
                                       but good for context).
            map_filename_to_subject_id (bool): If True, tries to extract a subject ID from the filename.
                                               Assumes filenames might contain patterns like 'sXXX_' or 'subjXXX_'.
                                               If False, uses default_subject_id.
            default_subject_id (int): Default subject ID to use if mapping is False or fails.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.filepaths = filepaths
        self.expected_channels = expected_channels
        self.expected_sfreq = expected_sfreq # Store this for potential future use or info
        self.model_patch_length = model_patch_length
        self.model_patch_stride = model_patch_stride
        self.map_filename_to_subject_id = map_filename_to_subject_id
        self.default_subject_id = default_subject_id
        self.transform = transform

        if not self.filepaths:
            raise ValueError("Filepaths list cannot be empty.")

        print(f"EEGPickleDataset initialized with {len(self.filepaths)} files.")
        if self.expected_sfreq:
            print(f"Dataset assumes data is at {self.expected_sfreq} Hz (LBLM paper uses 250Hz for its pretraining).")
            print("The TUH preprocessing script you provided resamples to 200Hz.")
            print("Ensure your data's sampling frequency matches what your model expects or was trained on.")


    def __len__(self):
        return len(self.filepaths)

    def _extract_subject_id_from_filename(self, filepath):
        """
        Placeholder function to extract subject ID.
        Customize this based on your filename patterns.
        Example: 'tuh_eeg_s001_t001_segment0.pkl' -> extract '001'
        """
        filename = Path(filepath).name
        # Try a few common patterns for subject IDs like s001, subj001, p001 etc.
        match = re.search(r'[sS]([0-9]+)', filename) # Looks for 's' or 'S' followed by numbers
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        match = re.search(r'subj([0-9]+)', filename, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        # Add more patterns if needed, e.g., for TUH specific subject naming if it's in filename root
        # For TUH, filenames are like '00000258_s002_t000_0.pkl' after your script.
        # Let's try to get the part after 's' if it exists and is numeric.
        parts = filename.split('_')
        for part in parts:
            if part.startswith('s') and len(part) > 1 and part[1:].isdigit():
                try:
                    return int(part[1:])
                except ValueError:
                    continue # Try next part if this 'sXXX' is not numeric

        # print(f"Warning: Could not extract subject ID from filename: {filename}. Using default.")
        return self.default_subject_id


    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        try:
            with open(filepath, 'rb') as f:
                data_dict = pickle.load(f)
        except Exception as e:
            raise IOError(f"Error loading pickle file {filepath}: {e}")
        eeg_data_np = data_dict.get("X")
        # ch_names = data_dict.get("ch_names") # Available if needed later

        if eeg_data_np is None:
            raise ValueError(f"Pickle file {filepath} does not contain 'X' key or it's None.")

        # Convert to float32, PyTorch default for nn.Linear etc.
        if isinstance(eeg_data_np, np.ndarray):
            eeg_data_np = eeg_data_np.astype(np.float32)
        elif isinstance(eeg_data_np, torch.Tensor):
            eeg_data_np = eeg_data_np.numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported data type: {type(eeg_data_np)}")

        # Basic checks
        if eeg_data_np.ndim != 2: # Should be [channels, timepoints]
            raise ValueError(f"EEG data in {filepath} has incorrect ndim: {eeg_data_np.ndim}. Expected 2.")

        num_loaded_channels, num_timepoints = eeg_data_np.shape

        if self.expected_channels is not None and num_loaded_channels != self.expected_channels:
            raise ValueError(f"EEG data in {filepath} has {num_loaded_channels} channels, "
                             f"but expected {self.expected_channels}. Shape is {eeg_data_np.shape}.")

        if num_timepoints < self.model_patch_length:
            raise ValueError(f"EEG data in {filepath} has {num_timepoints} timepoints, "
                             f"which is less than model_patch_length ({self.model_patch_length}). "
                             "Cannot create even one patch.")

        # EEG data should be [num_channels, timepoints]
        eeg_tensor = torch.from_numpy(eeg_data_np[:,:(self.model_patch_length-self.model_patch_stride)*100]) # Shape: [C, T]

        subject_id = self.default_subject_id
        if self.map_filename_to_subject_id:
            subject_id = self._extract_subject_id_from_filename(filepath)
        
        subject_id_tensor = torch.tensor(subject_id, dtype=torch.long)

        sample = {
            'x_raw_eeg': eeg_tensor/eeg_tensor.norm()*100,      # Model expects [B, C, T] - collate_fn will handle batch dim
            'subject_id': subject_id_tensor # Model expects [B]
        }

        if self.transform:
            sample = self.transform(sample)
            # Ensure transform maintains the expected keys and tensor types/shapes

        return sample

# Example usage (outside the class, for testing the dataset itself)
if __name__ == '__main__':
    print("Testing EEGPickleDataset...")

    # Create some dummy pickle files based on your script's output format
    dummy_data_dir = Path("./dummy_eeg_pickles")
    dummy_data_dir.mkdir(exist_ok=True)
    
    num_dummy_files = 5
    dummy_channels = 60 # Example, from TUH after dropping
    dummy_sfreq = 200   # From your script
    # time_length = (1024 // num_dummy_channels) * dummy_sfreq  # From your script
    # 1024 // 60 = 17.  17 * 200 = 3400 timepoints per segment
    dummy_timepoints_per_segment = 3400
    
    dummy_filepaths = []
    for i in range(num_dummy_files):
        # Example filename mimicking your script's output with a potential subject ID
        # Filename format: <original_edf_name_root>_s<subject_num>_t<session_num>_<segment_idx>.pkl
        # Or just <original_edf_name_root>_<segment_idx>.pkl
        # Let's try to include a subject pattern:
        subject_num_in_fname = i % 3 # 0, 1, 2, 0, 1
        filename = f"dummy_edf_s{subject_num_in_fname:03d}_segment_{i}.pkl"
        filepath = dummy_data_dir / filename
        
        dummy_eeg_data = np.random.randn(dummy_channels, dummy_timepoints_per_segment).astype(np.float32)
        dummy_ch_names = [f'CH{j}' for j in range(dummy_channels)]
        
        with open(filepath, 'wb') as f:
            pickle.dump({"X": dummy_eeg_data, "ch_names": dummy_ch_names}, f)
        dummy_filepaths.append(str(filepath))

    print(f"Created {len(dummy_filepaths)} dummy pickle files in {dummy_data_dir}")

    # --- Test Case 1: Basic instantiation ---
    print("\n--- Test Case 1: Basic ---")
    try:
        dataset = EEGPickleDataset(
            filepaths=dummy_filepaths,
            expected_channels=dummy_channels,
            expected_sfreq=dummy_sfreq, # LBLM uses 250Hz, your script uses 200Hz
            model_patch_length=25 # LBLM model patch length
        )
        print(f"Dataset length: {len(dataset)}")
        sample0 = dataset[0]
        print(f"Sample 0 keys: {sample0.keys()}")
        print(f"Sample 0 x_raw_eeg shape: {sample0['x_raw_eeg'].shape}") # Should be [C, T]
        print(f"Sample 0 subject_id: {sample0['subject_id']}")
        assert sample0['x_raw_eeg'].shape == (dummy_channels, dummy_timepoints_per_segment)
        assert sample0['x_raw_eeg'].dtype == torch.float32
        assert sample0['subject_id'].dtype == torch.long
    except Exception as e:
        print(f"Error in Test Case 1: {e}")
        import traceback
        traceback.print_exc()


    # --- Test Case 2: Subject ID mapping disabled ---
    print("\n--- Test Case 2: Subject ID mapping disabled ---")
    try:
        dataset_no_map = EEGPickleDataset(
            filepaths=dummy_filepaths,
            expected_channels=dummy_channels,
            model_patch_length=25,
            map_filename_to_subject_id=False,
            default_subject_id=99
        )
        sample1 = dataset_no_map[1]
        print(f"Sample 1 subject_id (no mapping): {sample1['subject_id']}")
        assert sample1['subject_id'].item() == 99
    except Exception as e:
        print(f"Error in Test Case 2: {e}")

    # --- Test Case 3: Insufficient timepoints ---
    print("\n--- Test Case 3: Insufficient timepoints ---")
    short_filepath = dummy_data_dir / "short_time_s000_segment_0.pkl"
    short_eeg_data = np.random.randn(dummy_channels, 10).astype(np.float32) # Only 10 timepoints
    with open(short_filepath, 'wb') as f:
        pickle.dump({"X": short_eeg_data, "ch_names": [f'CH{j}' for j in range(dummy_channels)]}, f)
    
    try:
        dataset_short = EEGPickleDataset(
            filepaths=[str(short_filepath)],
            expected_channels=dummy_channels,
            model_patch_length=25 # Expects at least 25 timepoints
        )
        _ = dataset_short[0] # This should raise ValueError
        print("Error: Test Case 3 did not raise ValueError for short timepoints as expected.")
    except ValueError as ve:
        print(f"Successfully caught expected ValueError in Test Case 3: {ve}")
    except Exception as e:
        print(f"Unexpected error in Test Case 3: {e}")


    # --- Test Case 4: Channel mismatch ---
    print("\n--- Test Case 4: Channel mismatch ---")
    try:
        dataset_ch_mismatch = EEGPickleDataset(
            filepaths=dummy_filepaths,
            expected_channels=dummy_channels - 1, # Expect one less channel
            model_patch_length=25
        )
        _ = dataset_ch_mismatch[0] # This should raise ValueError
        print("Error: Test Case 4 did not raise ValueError for channel mismatch as expected.")
    except ValueError as ve:
        print(f"Successfully caught expected ValueError in Test Case 4: {ve}")
    except Exception as e:
        print(f"Unexpected error in Test Case 4: {e}")

    print("\nEEGPickleDataset testing finished.")
    # Clean up dummy files (optional)
    # import shutil
    # shutil.rmtree(dummy_data_dir)
    # print(f"Cleaned up {dummy_data_dir}")


"""
from torch.utils.data import DataLoader
import glob

# 1. Get a list of your pickle filepaths
#    Replace '/path/to/your/pickles/' with the actual path to 'tuh_full'
pickle_files_pattern = "/workspace/tuh_full/*.pkl" # From your script
all_pickle_filepaths = glob.glob(pickle_files_pattern)

if not all_pickle_filepaths:
    print(f"Warning: No pickle files found at {pickle_files_pattern}. Dataset will be empty.")
    # Handle this case, maybe by exiting or creating dummy data for testing.

# 2. Instantiate the Dataset
#    Adjust expected_channels and expected_sfreq as per your data.
#    LBLM model expects eeg_channels (e.g., 122 from their paper).
#    Your TUH script uses rsfreq=200. If LBLM used 250Hz, this is a mismatch.
dataset_params = {
    "filepaths": all_pickle_filepaths,
    "expected_channels": 60, # Example: Number of channels AFTER your preprocessing
    "expected_sfreq": 200,   # Your TUH script output sfreq
    "model_patch_length": 25, # From your LBLM model config
    "map_filename_to_subject_id": True, # Or False if you want to use default
    "default_subject_id": 0
}
eeg_dataset = EEGPickleDataset(**dataset_params)

# 3. Instantiate the DataLoader
if len(eeg_dataset) > 0:
    train_dataloader = DataLoader(eeg_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Example of iterating through the dataloader
    # for batch in train_dataloader:
    #     x_eeg_batch = batch['x_raw_eeg']  # Shape: [batch_size, num_channels, timepoints]
    #     subj_id_batch = batch['subject_id'] # Shape: [batch_size]
    #     # ... feed to your model ...
    #     # print(x_eeg_batch.shape, subj_id_batch.shape)
    #     break # Just show one batch
else:
    print("Dataset is empty, DataLoader not created.")
"""