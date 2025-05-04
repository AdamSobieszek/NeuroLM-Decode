"""

Code for transforming the fif file which is created from the LSL data and initially filtered into a tensor along with metadata in pandas.


"""

import mne
import numpy as np
import torch
import pandas as pd
import json
from typing import Tuple, Union, List
import ast
from src.data_utils.fif_to_fif_processing import process_fif_to_fif

# for mne complaining about file names
import warnings

def single_fif_to_tensor(single_fif: mne.io.Raw, 
                         #n_repetitions: int = 10,
                         tmin: float = 0,
                         tmax: float = 1,
                         baseline: Tuple[float, float] = (0,0),
                         verbose: bool = False) -> Tuple[torch.Tensor, pd.DataFrame]:

    """
    Processes a single fif object.

    Extracts the epoch data and then reshapes it into:
    [word, repetitions within a word, channels, time]
    
    Also prepares the pandas DataFrame with metadata.
    
    DataFrame shape: each rows corresponds to a single word group of repetitions,
    columns contains metadata.

    Parameters:
        single_fif (mne.io.Raw): The fif object to process.
        n_repetitions (int): Number of repetitions to average. Default is 10.
        tmin (float): The start time of the epochs. Default is 0.
        tmax (float): The end time of the epochs. Default is 1.
        baseline (tuple): The baseline to apply to the epochs. Default is (0, 0).
        verbose (bool): Verbosity level. Default is False.

    Returns:
        Tuple[torch.Tensor, pd.DataFrame]: The processed data tensor and the metadata DataFrame.

    """

    single_fif = single_fif.copy()

    n_repetitions = string_to_dictionary(single_fif.info['description'])['experiment_description']['n_repetitions']

    events, event_id = mne.events_from_annotations(single_fif, verbose=verbose)
    epochs = mne.Epochs(single_fif, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=baseline, verbose=verbose)
    epochs_data = epochs.get_data(verbose=verbose)

    epochs_data = epochs_data[:,:,1:] # seems that mne.Epochs add a leading zero

    n_channels = epochs_data.shape[1]
    n_samples = epochs_data.shape[2]

    # reshaping into [word, repetitions within a word, channels, time]
    eeg_data = epochs_data.reshape(-1, n_repetitions, n_channels, n_samples)
    eeg_data = torch.Tensor(eeg_data)

    event_id_to_word = {v: k for k, v in event_id.items()}
    words = epochs.events[:, 2]
    words = np.array([event_id_to_word[word] for word in words])
    words = words.reshape(-1, n_repetitions)
    words = words[:,0]

    fif_description_dict = string_to_dictionary(single_fif.info['description'])
    if 'processing_history' in fif_description_dict:
        processing_dict = fif_description_dict['processing_history']
    else:
        processing_dict = None

    #processing_dict = string_to_dictionary(single_fif.info['description']['processing_history'])
    #if processing_dict is not None:
        #processing_dict = json.loads(json.dumps(eval(processing_dict)))


    df = pd.DataFrame({
        'word': words,
        'sfreq': [single_fif.info['sfreq']] * len(words),
        'measurement_date': [single_fif.info['meas_date']] * len(words),
        'n_repetitions': [n_repetitions] * len(words),
        'ch_names': [single_fif.ch_names] * len(words),
        'first_processing_description': [processing_dict] * len(words), # this will be improved
        })

    return eeg_data, df

def string_to_dictionary(dict_string):
    """
    Converts a string representation of a dictionary back to a Python dictionary.
    Uses ast.literal_eval for safe evaluation.

    Args:
        dict_string (str): The string representation of a dictionary.

    Returns:
        dict: The reconstructed dictionary.
        Returns None if the string cannot be safely evaluated as a dictionary.
    """
    try:
        # Use ast.literal_eval for safe evaluation of string as a Python literal
        return ast.literal_eval(dict_string)
    except (ValueError, SyntaxError):
        print("Error: Could not safely convert string to dictionary. "
              "Ensure the string is a valid dictionary representation.")
        return None

def concatenate_and_crop_tensors(tensor_list, verbose):
    """
    Concatenates a list of tensors along the first dimension after cropping
    their second dimensions to the minimum length among all tensors.

    Args:
        tensor_list (list of torch.Tensor): A list of tensors with potentially
                                            different lengths in the second dimension.

    Returns:
        torch.Tensor: The concatenated tensor, or None if the input list is empty.
    """
    if not tensor_list:
        return None  # Handle empty list case

    min_dim2 = float('inf')  # Initialize with infinity to find the minimum

    # 1. Find the minimum length of the second dimension
    for tensor in tensor_list:
        if tensor.ndim < 2:  # Ensure tensor has at least 2 dimensions
            raise ValueError("Tensors must have at least 2 dimensions.")
        min_dim2 = min(min_dim2, tensor.shape[1])

    if min_dim2 == float('inf'): # if no tensors in the list or all have dim < 2
        return None

    cropped_tensors = []
    # 2. Crop tensors to the minimum second dimension length
    for tensor in tensor_list:
        cropped_tensor = tensor[:, :min_dim2]
        if cropped_tensor.shape[1] != tensor.shape[1]:
            if verbose:
                print(f"Cropping tensor from {tensor.shape} to {cropped_tensor.shape}")
        cropped_tensors.append(cropped_tensor)

    # 3. Concatenate the cropped tensors along the first dimension (dim=0)
    concatenated_tensor = torch.cat(cropped_tensors, dim=0)

    return concatenated_tensor

def transform_fif_to_tensor(eeg_data: Union[str,
                                            List[str],
                                            mne.io.Raw, 
                                            List[mne.io.Raw]
                                            ],
                            processing_dict: Union[dict, str] = None,
                            tmin: float = 0,
                            tmax: float = 1,
                            baseline: Tuple[float, float] = (0,0),
                            verbose = False) -> Tuple[torch.Tensor, pd.DataFrame]:


    if processing_dict is not None:
        eeg_data, processing_dict = process_fif_to_fif(eeg_data=eeg_data, processing_dict=processing_dict)

    if not isinstance(eeg_data, list):
        eeg_data = [eeg_data]

    assert isinstance(eeg_data[0], (mne.io.Raw, mne.io.array.RawArray, mne.io.array.RawArray, str)), 'eeg_data should be given as a list of mne.io.Raw or str paths to fif files'

    if isinstance(eeg_data[0], str):
        eeg_data = [mne.io.read_raw_fif(eeg_file, verbose=verbose) for eeg_file in eeg_data]

    if verbose == False:
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    eeg_tensors = []
    all_dfs = []

    for eeg_recording in eeg_data:
        
        eeg_tensor, df = single_fif_to_tensor(
            single_fif=eeg_recording,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            verbose=verbose)
        
        eeg_tensors.append(eeg_tensor) 
        all_dfs.append(df)

    try:
        #eeg_tensor = torch.cat(eeg_tensors, dim=0)
        eeg_tensor = concatenate_and_crop_tensors(eeg_tensors, verbose=verbose)
    except Exception as e:
        print(e)
        raise Exception('Error concatenating eeg tensors from multiple recordings')
    
    combined_df = pd.concat(all_dfs, ignore_index=True)

    if processing_dict is not None:
        for key, value in processing_dict.items():
            combined_df[key] = [value] * len(combined_df)

    return eeg_tensor, combined_df


"""

Older function with was rejecting epochs using autoreject.
It will require reimplementing it for tensors.
Which will require substituting the epochs which were rejected with average of the adjacent ones.

"""

# def autoreject_and_stack(
#     raw_objects: Union[mne.io.Raw, List[mne.io.Raw]],
#     drop_channels: Optional[List[str]] = None,
#     baseline: Tuple[float, float] = (-0.05, 0),
#     tmin: float = -0.05,
#     tmax: float = None,
#     n_jobs: int = 1,
#     verbose: bool = False,
#     apply_autoreject = False
#     ) -> mne.Epochs:
#     """
#     Applies Autoreject artifact rejection to one or more Raw EEG objects and stacks the resulting epochs.
#     Uses the shortest valid event duration across all files for epoching.
#     Assumes all Raw objects have the same event IDs defined in their annotations.

#     Args:
#         raw_objects: A single mne.io.Raw object or a list of mne.io.Raw objects.
#         drop_channels: An optional list of channel names to drop before epoching.
#         n_jobs: Number of jobs to run in parallel for Autoreject.

#     Returns:
#         A single mne.Epochs object containing the cleaned and stacked epochs.
#     """

#     if isinstance(raw_objects, list):
#         raw_list = raw_objects
#     else:
#         raw_list = [raw_objects]

#     if not raw_list:
#         raise ValueError("No Raw objects provided.")

#     all_epochs = []
#     min_duration = float('inf')
#     consistent_event_id = None

#     for raw in raw_list:
#         events, event_dict_current = mne.events_from_annotations(raw)
#         if consistent_event_id is None:
#             consistent_event_id = event_dict_current
#         elif consistent_event_id != event_dict_current:
#             raise ValueError("Event IDs are not consistent across all Raw objects.")

#     if consistent_event_id is None:
#         raise ValueError("No events found in the provided Raw objects.")

#     if tmax is None:
#         min_duration = float('inf')
#         # Find the shortest valid event duration
#         for raw in raw_list:
#             if drop_channels:
#                 raw.drop_channels(drop_channels, on_missing='ignore')

#             events, _ = mne.events_from_annotations(raw)
#             annotations = raw.annotations

#             for event_idx, event in enumerate(events):
#                 annotation_desc = annotations[event_idx]['description']
#                 if annotation_desc in consistent_event_id:
#                     duration = annotations[event_idx]['duration']
#                     if duration > 0:
#                         min_duration = min(min_duration, duration)

#         if min_duration == float('inf'):
#             raise ValueError("No valid event durations found across the provided Raw objects.")

#         print(f"Using shortest event duration: {min_duration:.3f} seconds for epoching.")

#     for raw in raw_list:
#         if drop_channels:
#             raw.drop_channels(drop_channels, on_missing='ignore')

#         events, _ = mne.events_from_annotations(raw)

#         # Epoch the raw data using the determined minimum duration
#         current_epochs = mne.Epochs(
#             raw,
#             events=events,
#             event_id=consistent_event_id,
#             tmin=tmin,
#             tmax=min_duration,
#             baseline=baseline,
#             preload=True,
#             on_missing='ignore',
#             verbose=verbose
#         )

#         if apply_autoreject:
#             # Apply Autoreject
#             ar = autoreject.AutoReject(n_jobs=n_jobs)
#             cleaned_epochs = ar.fit_transform(current_epochs)
#             print(f"Applied Autoreject. Dropped {len(current_epochs) - len(cleaned_epochs)} epochs.")

#             all_epochs.append(cleaned_epochs)
#         else:
#             all_epochs.append(current_epochs)

#     """It seems that when stacking fifs from different experiments the code below might not be needed - to be confirmed"""

#     # # Stack the cleaned epochs
#     # all_annotations = mne.Annotations([], [], [])
#     # offset = 0
#     # for epochs in all_epochs:
#     #     if epochs.annotations:
#     #         annots = epochs.annotations.copy()
#     #         annots.onset += offset
#     #         all_annotations = all_annotations + annots
#     #     offset += len(epochs.times) / epochs.info['sfreq']

#     stacked_epochs = mne.concatenate_epochs(all_epochs, on_mismatch='warn')
#     #stacked_epochs.set_annotations(all_annotations) 
#     print("Stacked all cleaned epochs into a single Epochs object.")
#     return stacked_epochs