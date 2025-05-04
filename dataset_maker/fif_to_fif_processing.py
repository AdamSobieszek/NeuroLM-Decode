from typing import List, Union, Tuple
import mne
import warnings
import yaml
import ast


# takes a fif file and a dictionary with keys being the method names and values - dictionaries for these method parameters

def process_fif_to_fif(
        eeg_data: Union[str, mne.io.Raw, List[mne.io.Raw], List[str]],
        processing_dict: Union[dict, str],
        verbose = False,
        ) -> Tuple[Union[mne.io.Raw, List[mne.io.Raw]], dict]:

    if not isinstance(eeg_data, list):
        eeg_data = [eeg_data]

    assert isinstance(eeg_data[0], (mne.io.Raw, mne.io.array.RawArray, mne.io.array.RawArray, str)), 'eeg_data should be given as a list of mne.io.Raw or str paths to fif files'

    if isinstance(eeg_data[0], str):
        eeg_data = [mne.io.read_raw_fif(eeg_file, verbose=verbose) for eeg_file in eeg_data]

    if isinstance(processing_dict, str):
        processing_dict = yaml.safe_load(open(processing_dict, 'r'))

    if verbose == False:
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    processed_eeg_data = []
    for eeg_recording in eeg_data:
        processed_eeg_recording = eeg_recording.copy()
        processed_eeg_recording.load_data(verbose=verbose) # mne by default doesn't load the data, this makes sure that preload=True is not necessary when passing fif
        for processing_function_name, processing_function_params in processing_dict.items():
            if processing_function_name in globals(): #check if func exists
                processing_function = globals()[processing_function_name]
                processed_eeg_recording = processing_function(processed_eeg_recording, verbose=verbose, **processing_function_params)
                #processed_eeg_recording = method_name(eeg_recording, **method_params)
            else:
                raise ValueError(f"Processing function '{processing_function_name}' not found.")
        processed_eeg_data.append(processed_eeg_recording)
        append_fif_proc_history(processed_eeg_recording, processing_dict)

    if len(eeg_data) == 1:
        eeg_data = processed_eeg_data[0]
    else:
        eeg_data = processed_eeg_data

    #append_fif_proc_history(eeg_data, processing_dict)

    return eeg_data, processing_dict

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

def append_fif_proc_history(fif, processing_dict):
    
    """Appends the processing history to the fif file."""
    
    # if 'proc_history' not in fif.info:
    #     fif.info['proc_history'] = []

    description_dict = string_to_dictionary(fif.info['description'])
    if 'processing_history' not in description_dict.keys():
        description_dict['processing_history'] = []
        
    description_dict['processing_history'].append(processing_dict)
    fif.info['description'] = str(description_dict)
    
    return fif

def low_high_pass_filter(raw, l_freq: int = 1, h_freq: int = None, verbose=False, **kwargs) -> mne.io.RawArray:
    """Filters the Raw object."""

    default_params = {
        'l_freq': l_freq,
        'h_freq': h_freq
    }

    params = default_params.copy()
    params.update(kwargs)

    raw_filtered = raw.copy().filter(verbose=verbose, **params)
    return raw_filtered

def downsample(raw, sfreq: int = 250, verbose=False, **kwargs) -> mne.io.RawArray:
    """Downsamples the Raw object."""
    
    default_params = {
        'sfreq': sfreq,
    }

    params = default_params.copy()
    params.update(kwargs)

    raw_downsampled = raw.copy().resample(verbose=verbose, **params)
    return raw_downsampled

def notch_filter(raw, freqs: list = [50, 100], verbose=False, filter_length='auto', phase='zero', method='fir', **kwargs) -> mne.io.RawArray:

    """Notches the Raw object."""
    default_params = {
        'freqs': freqs,
        'filter_length': filter_length,
        'phase': phase,
        'method': method,
    }

    params = default_params.copy()  # Ensure defaults are not modified
    params.update(kwargs)

    raw_notched = raw.copy().notch_filter(verbose=verbose, **params)
    return raw_notched

def average_reference(eeg_signal: mne.io.RawArray, verbose = False, **kwargs) -> mne.io.RawArray:
    """
    Apply average reference to the EEG signal.
    """
    return mne.set_eeg_reference(eeg_signal, ref_channels='average', verbose=verbose, **kwargs)

def laplacian(eeg_signal: mne.io.RawArray, verbose = False, **kwargs) -> mne.io.RawArray:
    """
    Apply Laplacian reference to the EEG signal.
    """
    return mne.preprocessing.compute_current_source_density(eeg_signal, verbose=verbose, **kwargs)

def interpolate_channels(eeg_signal: mne.io.RawArray, bad_channels: List[str] = None, verbose: bool = False, reset_bads: bool = False, **kwargs) -> mne.io.RawArray:
    """

    Interpolate bad channels in the EEG signal.

    """

    raw_copy = eeg_signal.copy()

    default_params = {
        #'bads': bad_channels,
        'reset_bads': reset_bads,
        #'verbose': verbose,
    }

    params = default_params.copy()  # Ensure defaults are not modified
    params.update(kwargs)

    if bad_channels is not None:
        raw_copy.info['bads'] = bad_channels
    return raw_copy.interpolate_bads(verbose = verbose, **params)

def custom_reference(eeg_signal: mne.io.RawArray, verbose: bool = False, **kwargs) -> mne.io.RawArray:
    """
    Apply custom reference to the EEG signal.
    
    Parameters:
    - eeg_signal: The input EEG signal
    - ref_channels: List of channel names or indices to use as reference - ref_channels: List[str]
    
    Example:
    methods_and_params = {
        # ... other methods ...
        'custom_reference': {'ref_channels': ['Cz', 'Fz']}  # Example reference channels
    }
    """

    raw_copy = eeg_signal.copy()

    return raw_copy.set_eeg_reference(eeg_signal, verbose = verbose, **kwargs)


def bipolar_reference(eeg_signal: mne.io.RawArray, verbose: bool = False, **kwargs) -> mne.io.RawArray:
    """
    Applies bipolar referencing to an MNE RawArray object, replacing channels in-place.

    Parameters:
        raw (mne.io.RawArray): The input EEG signal. Modified in-place.
        channel_pairs (List[Tuple[str, str]]): A list of tuples, each containing two channel names to subtract.
            The first channel is referenced to the second, overwriting the first channel. 
        kwargs: Optional keyword arguments (currently not used, but kept for flexibility).

    Returns:
        mne.io.RawArray: The RawArray with bipolar referenced channels, original channels are replaced.
    """

    raw_copy = eeg_signal.copy()

    channel_pairs = kwargs.get('channel_pairs')

    bipolar_data = []  # Store bipolar data
    bipolar_ch_names = [] # Store bipolar channel names
    channels_to_replace = [] # Store the name of the channel that will be replaced

    for ch1, ch2 in channel_pairs:
        try:
            ch1_idx = raw_copy.ch_names.index(ch1)
            ch2_idx = raw_copy.ch_names.index(ch2)
            bipolar_data.append(raw_copy.get_data(picks=[ch1_idx]) - raw_copy.get_data(picks=[ch2_idx]))
            bipolar_ch_names.append(ch1)  # Keep original channel name
            channels_to_replace.append(ch1)
        except ValueError:
             print(f"Warning: Could not apply bipolar reference using {ch1} and {ch2} because one or both of the channels were not present.")


    # Replace original channels with bipolar data
    for i, ch_name in enumerate(channels_to_replace):
      ch_idx = raw_copy.ch_names.index(ch_name)
      raw_copy._data[ch_idx] = bipolar_data[i].flatten() # replace data for the given channel

    return raw_copy

def ica(eeg_signal: mne.io.RawArray, verbose: bool = False, method='fastica', **kwargs) -> mne.io.RawArray:
    """
    Applies ICA to an MNE RawArray object in-place.

    Parameters:
        raw (mne.io.RawArray): The input EEG signal. Modified in-place.
        verbose (bool): Controls verbosity
        kwargs: Keyword arguments to be passed to the mne.preprocessing.ICA constructor.
            Common arguments would include 'n_components', 'random_state', 'method'.

    Returns:
        mne.io.RawArray: The RawArray with ICA applied in-place.
    """

    raw_copy = eeg_signal.copy()

    default_params = {
        'method': method
    }

    params = default_params.copy()  # Ensure defaults are not modified
    params.update(kwargs)
    
    try:
      ica = mne.preprocessing.ICA(verbose = verbose, **params)
      ica.fit(raw_copy, verbose=verbose)
      ica.apply(raw_copy, verbose=verbose)
    except Exception as e:
      print(f"Warning: Could not apply ICA due to {e}")
    return raw_copy