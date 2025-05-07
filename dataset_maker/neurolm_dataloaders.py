"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

from torch.utils.data import Dataset
import torch
import warnings
import mne
from tqdm import tqdm
import numpy as np
import scipy
from typing import Union, List

from src.data_utils.fif_to_fif_processing import process_fif_to_fif
from src.data_utils.fif_to_tensor_processing import string_to_dictionary
from src.data_utils.fif_to_tensor_processing import transform_fif_to_tensor
#import pickle

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", \
    'pad', 'I1', 'I2'
]

"""
TODO:
* fix the words required by the fif_to_tensor_processing
"""

class FifLoader(Dataset):
    def __init__(self, 
                 eeg_data: Union[str,
                    List[str],
                    mne.io.Raw,
                    List[mne.io.Raw]
                    ],
                 block_size=1024,
                 extract_all_blocks=False, # if True, the fif files will be split into windows of a given length regardless of the events
                 window_length_s=1, # the length of the windows in seconds
                 processing_dict: Union[dict, str] = None,
                 scaling_factor: float = 2e6,
                 GPT_training=False,
                 verbose=False):
        
        self.eeg_data = eeg_data
        self.default_rate = 200
        self.window_length_s = window_length_s
        #self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.GPT_training = GPT_training
        self.verbose = verbose

        if self.verbose:
            print(f"Loading {len(self.files)} fif files")

        if not isinstance(eeg_data, list):
            eeg_data = [eeg_data]

        assert isinstance(eeg_data[0], (mne.io.Raw, mne.io.array.RawArray, mne.io.array.RawArray, str)), 'eeg_data should be given as a list of mne.io.Raw or str paths to fif files'

        if isinstance(eeg_data[0], str):
            eeg_data = [mne.io.read_raw(eeg_file, verbose=verbose, preload=True) for eeg_file in eeg_data]

        if self.verbose:
            print(f"Loaded {len(eeg_data)} fif files")
            fif_lengths = sum([fif.n_times for fif in eeg_data]) / self.default_rate
            print(f"Total length of fif files: {fif_lengths} seconds")

        if processing_dict is not None:
            if isinstance(processing_dict, str):
                processing_dict = string_to_dictionary(processing_dict)
            if self.verbose:
                print(f"Processing fif files with the following dictionary: {processing_dict}")
            eeg_data, processing_dict = process_fif_to_fif(eeg_data=eeg_data, processing_dict=processing_dict)

        # resampling fifs to the default rate
        if eeg_data[0].info['sfreq'] != self.default_rate:
            if self.verbose:
                print(f"Resampling fifs to the default rate of {self.default_rate} Hz")
            eeg_data = [fif.resample(self.default_rate) for fif in eeg_data]
        
        assert [fif.ch_names for fif in eeg_data] == [eeg_data[0].ch_names], "All fif files should have the same ch_names for now, we'll improve it later"
        self.ch_names = eeg_data[0].ch_names

        # till this point eeg_data is a list of mne.io.Raw objects
        if extract_all_blocks:
            self.all_blocks = self._split_fifs_into_blocks(eeg_data)
        else:
            tensors, metadata_df = transform_fif_to_tensor(eeg_data, processing_dict, verbose=verbose)
            self.all_blocks = tensors
        
        # data collected from the perun LSL doesn't land in the correct scale, so this seems to rescale them somewhat correctly into uV
        self.all_blocks = self.all_blocks * scaling_factor

    def _split_single_fif_into_blocks(self, fif):
        """
        Finds the first and last event in the fif file and splits the fif file into blocks of a given length.
        """

        first_annotation_time = fif.annotations[0]['onset']
        last_annotation_time = fif.annotations[-1]['onset']

        #crop the fif file to the first and last event
        fif = fif.copy().crop(first_annotation_time, last_annotation_time)

        #split the fif file into blocks of a given length
        blocks = []
        fif_np = fif.get_data()
        block_length = self.window_length_s * self.default_rate
        
        for i in range(0, fif_np.shape[1], block_length):
            # Skip the last block if it's too short
            if i + block_length > fif_np.shape[1]:
                if self.verbose:
                    print(f"Dropping last block of length {fif_np.shape[1] - i} samples (shorter than {block_length} samples)")
                break
            blocks.append(fif_np[:, i:i+block_length])

        return torch.FloatTensor(blocks)

    def _split_fifs_into_blocks(self, eeg_data):
        """
        Splits the fif files into blocks of a given length.
        """
        blocks = []
        for fif in eeg_data:
            blocks.extend(self._split_single_fif_into_blocks(fif))
        
        # change the list of [fif, repetitions, channels, time] into a tensor of shape [repetitions, channels, time]
        blocks = torch.stack(blocks, dim=0)
        
        return blocks


    def __len__(self):
        return len(self.all_blocks)

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def get_chans(self, ch_names):
            chans = []
            for ch_name in ch_names:
                try:
                    chans.append(standard_1020.index(ch_name))
                except ValueError:
                    raise ValueError(f"Channel {ch_name} not found in standard_1020")
            return chans

    def __getitem__(self, index):
        #sample = pickle.load(open(self.files[index], "rb"))
        #data = sample["X"]
        #ch_names = sample["ch_names"]
        #data = torch.FloatTensor(data / 100)

        data = self.all_blocks[index]

        # splits the data into windows which are then indexed with the learnable embedding?
        time = data.size(1) // 200
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        #data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        data = data.view(data.size(0), -1, 200) # Reshape to [NumChannels, NumPatches, PatchLength]
        data = data.permute(1, 0, 2) # Reshape to [NumPatches, NumChannels, PatchLength]
        data = data.contiguous().view(-1, 200) # Reshape to [NumPatches * NumChannels, PatchLength]
        # this creates a data in a format 
        # [ch_1_patch_1, ch_2_patch_1, ch_3_patch_1, ..., ch_1_patch_2, ch_2_patch_2, ch_3_patch_2, ...]
        
        X = torch.zeros((self.block_size, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100))
            Y_raw = torch.zeros((self.block_size, 200))
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)
        
        # input_chans is the indices of the channels in the standard_1020 list
        # used for the spatial embedding
        input_chans = list(self.ch_names) * time
        input_chans.extend(['pad'] * (self.block_size - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        # input_time is the mask for padding zeros
        # ensure that padding zeros are not used in the attention mechanism
        input_time.extend([0] * (self.block_size - data.size(0)))
        input_time = torch.IntTensor(input_time)

        input_mask = torch.ones(self.block_size)
        input_mask[data.size(0):] = 0

        if self.GPT_training:
            # # gpt_mask is the mask for the GPT model
            # gpt_mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1, self.block_size, self.block_size)
            # num_chans = len(ch_names)
            # for i in range(time):
            #     gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1

            # gpt_mask[:, :, num_chans * time:] = 0
            # return X, input_chans, input_time, input_mask.bool(), gpt_mask.bool(), num_chans, data.size(0)
            warnings.warn("GPT training is not implemented yet")

        return X, Y_freq, Y_raw, input_chans, input_time, input_mask.bool()
    


# base_1005_to_biosemi = {
#     # Midline
#     'Fpz': 'A5', 'AFz': 'A3', 'Fz':  'A1', 'FCz': 'A2', 'Cz':  'B1',
#     'CPz': 'B2', 'Pz':  'C15', 'POz': 'C16', 'Oz':  'C22', 'Iz':  'C24',
#     # Left Hemisphere
#     'Fp1': 'A8', 'AF7': 'A10', 'AF3': 'A7', 'F7':  'A11', 'F5':  'A9',
#     'F3':  'A6', 'F1':  'A4', 'FT7': 'B10', 'FC5': 'B7', 'FC3': 'B5',
#     'FC1': 'B4', 'T7':  'B8', 'C5':  'B9', 'C3':  'B6', 'C1':  'B3',
#     'TP7': 'B14', 'CP5': 'C6', 'CP3': 'C14', 'CP1': 'C1', 'P7':  'C8',
#     'P5':  'C7', 'P3':  'C13', 'P1':  'C11', 'PO7': 'C9', 'PO3': 'C12',
#     'O1':  'C20',
#     # Right Hemisphere
#     'Fp2': 'A21', 'AF8': 'A19', 'AF4': 'A22', 'F8':  'A20', 'F6':  'A24',
#     'F4':  'A23', 'F2':  'A25', 'FT8': 'B25', 'FC6': 'B22', 'FC4': 'B20',
#     'FC2': 'B19', 'T8':  'B23', 'C6':  'B24', 'C4':  'B21', 'C2':  'B18',
#     'TP8': 'B29', 'CP6': 'C19', 'CP4': 'C27', 'CP2': 'C2', 'P8':  'C18',
#     'P6':  'C17', 'P4':  'C26', 'P2':  'C30', 'PO8': 'C21', 'PO4': 'C28',
#     'O2':  'C29',
# }
# biosemi_to_1005_map = {v: k for k, v in base_1005_to_biosemi.items()}


class ThinkingOutLoudLoader():
    def __init__(self,
                 eeg_data: Union[str,
                    List[str],
                    mne.io.Raw, 
                    List[mne.io.Raw]
                    ], 
                 block_size=1024,
                 extract_all_blocks=False, # if True, the fif files will be split into windows of a given length regardless of the events
                 window_length_s=1, # the length of the windows in seconds
                 processing_dict: Union[dict, str] = None,
                 GPT_training=False,
                 verbose=False
                 ):
        
        """
        Data is filtered as for our data - 1hz-inf, 50hz notch filter
        """

        self.base_1005_to_biosemi = {
            # Midline
            'Fpz': 'A5', 'AFz': 'A3', 'Fz':  'A1', 'FCz': 'A2', 'Cz':  'B1',
            'CPz': 'B2', 'Pz':  'C15', 'POz': 'C16', 'Oz':  'C22', 'Iz':  'C24',
            # Left Hemisphere
            'Fp1': 'A8', 'AF7': 'A10', 'AF3': 'A7', 'F7':  'A11', 'F5':  'A9',
            'F3':  'A6', 'F1':  'A4', 'FT7': 'B10', 'FC5': 'B7', 'FC3': 'B5',
            'FC1': 'B4', 'T7':  'B8', 'C5':  'B9', 'C3':  'B6', 'C1':  'B3',
            'TP7': 'B14', 'CP5': 'C6', 'CP3': 'C14', 'CP1': 'C1', 'P7':  'C8',
            'P5':  'C7', 'P3':  'C13', 'P1':  'C11', 'PO7': 'C9', 'PO3': 'C12',
            'O1':  'C20',
            # Right Hemisphere
            'Fp2': 'A21', 'AF8': 'A19', 'AF4': 'A22', 'F8':  'A20', 'F6':  'A24',
            'F4':  'A23', 'F2':  'A25', 'FT8': 'B25', 'FC6': 'B22', 'FC4': 'B20',
            'FC2': 'B19', 'T8':  'B23', 'C6':  'B24', 'C4':  'B21', 'C2':  'B18',
            'TP8': 'B29', 'CP6': 'C19', 'CP4': 'C27', 'CP2': 'C2', 'P8':  'C18',
            'P6':  'C17', 'P4':  'C26', 'P2':  'C30', 'PO8': 'C21', 'PO4': 'C28',
            'O2':  'C29',
        }
        self.biosemi_to_1005_map = {v: k for k, v in self.base_1005_to_biosemi.items()}

        self.eeg_data = eeg_data
        self.default_rate = 200
        self.window_length_s = window_length_s
        #self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.GPT_training = GPT_training
        self.verbose = verbose


        if not isinstance(eeg_data, list):
            eeg_data = [eeg_data]

        if self.verbose:
            print(f"Loading {len(self.eeg_data)} bdf files")

        assert isinstance(eeg_data[0], (mne.io.Raw, mne.io.array.RawArray, mne.io.array.RawArray, str)), 'eeg_data should be given as a list of mne.io.Raw or str paths to bdf files'

        if isinstance(eeg_data[0], str):
            raw_eeg_data = [mne.io.read_raw(recording_path, verbose=verbose, preload=True) for recording_path in eeg_data]
        else:
            raw_eeg_data = eeg_data

        if self.verbose:
            progress_bar = tqdm(raw_eeg_data, desc="Processing Thinking Out Loud Data")
        else:
            progress_bar = raw_eeg_data

        eeg_data = []
        for recording in progress_bar:
            eeg_data.append(self._process_thinking_out_loud_bdf(recording))

        if processing_dict is not None:
            if isinstance(processing_dict, str):
                processing_dict = string_to_dictionary(processing_dict)
            if self.verbose:
                print(f"Processing fif files with the following dictionary: {processing_dict}")
            eeg_data, processing_dict = process_fif_to_fif(eeg_data=eeg_data, processing_dict=processing_dict)

        self.ch_names = [x.upper() for x in list(self.base_1005_to_biosemi.keys())]

        # Extracting data from the processed bdf files
        if extract_all_blocks:
            self.all_blocks = self._split_bdfs_into_all_blocks(eeg_data)
        else:
            self.all_blocks = self._split_bdfs_into_torch_epochs(eeg_data)

    def __len__(self):
        return len(self.all_blocks)

    def _process_thinking_out_loud_bdf(self, bdf):
        """
        Processes Thinking Out Loud bdf files by:
        * low and hig pass filtering
        * applying notch filter at 50Hz
        * resampling to 200Hz
        * setting the reference channel
        * adapting the montage from biosemi128 to 1020
        * removing the unnecessary channels
        """

        reference_channels = ['EXG1', 'EXG2']
        low_cut = 1
        high_cut = None
        aquisition_eq = "biosemi128"
        channels_to_drop = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
        
        bdf = bdf.copy()
        bdf.set_eeg_reference(ref_channels=reference_channels, verbose=self.verbose)
        bdf.filter(l_freq=low_cut, h_freq=high_cut, verbose=self.verbose)
        bdf.notch_filter(freqs=50, verbose=self.verbose)

        # setting events
        events = mne.find_events(bdf, initial_event=True,
                                        consecutive=True, min_duration=0.002, verbose=self.verbose)
        #bdf.events = events
        # Create annotations from events
        # Use the events that were already found above
        event_id = {}
        for event_code in np.unique(events[:, 2]):
            event_id[f'event_{event_code}'] = event_code
            
        # Convert events to annotations
        onsets = events[:, 0] / bdf.info['sfreq']  # Convert from samples to seconds
        durations = np.zeros_like(onsets)  # Instantaneous events
        descriptions = [code for code in events[:, 2]]
        
        # Create annotations object
        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=bdf.info['meas_date'],
        )
        
        # Set annotations to the raw object
        bdf.set_annotations(annotations)
        bdf.resample(self.default_rate, verbose=self.verbose)

        bdf.drop_channels(channels_to_drop)

        for ch in bdf.ch_names:
            if ch not in self.biosemi_to_1005_map:
                #dropping channels that are not in the biosemi_to_1005_map
                bdf.drop_channels([ch])

        # renaming the channels to the standard 1020 montage
        bdf.rename_channels(self.biosemi_to_1005_map, verbose=self.verbose)
        bdf.set_montage(mne.channels.make_standard_montage('standard_1020'), verbose=self.verbose)

        return bdf   

    def _split_single_bdf_into_all_blocks(self, bdf):
        """
        Finds the first and last event in the fif file and splits the fif file into blocks of a given length.
        """

        # the first and last annotations seems to be the start and stop events, which we don't want to include in the data
        first_annotation_timepoint = bdf.annotations[0]['onset']
        last_annotation_timepoint = bdf.annotations[-1]['onset']

        #crop the bdf file to the first and last event
        bdf = bdf.copy().crop(first_annotation_timepoint, last_annotation_timepoint)

        #split the fif file into blocks of a given length
        blocks = []
        bdf_np = bdf.get_data()
        block_length = self.window_length_s * self.default_rate
        
        for i in range(0, bdf_np.shape[1], block_length):
            # Skip the last block if it's too short
            if i + block_length > bdf_np.shape[1]:
                if self.verbose:
                    print(f"Dropping last block of length {bdf_np.shape[1] - i} samples (shorter than {block_length} samples)")
                break
            blocks.append(bdf_np[:, i:i+block_length])
        
        blocks_np = np.array(blocks)#np.stack(blocks, axis=0)
        return torch.FloatTensor(blocks_np)

    def _split_bdfs_into_all_blocks(self, eeg_data):
        """
        Splits the fif files into blocks of a given length.
        """
        blocks = []
        if self.verbose:
            progress_bar = tqdm(eeg_data, desc="Splitting bdfs into blocks")
        else:
            progress_bar = eeg_data

        for bdf in progress_bar:
            blocks.extend(self._split_single_bdf_into_all_blocks(bdf))
        
        # change the list of [fif, repetitions, channels, time] into a tensor of shape [repetitions, channels, time]
        blocks = torch.stack(blocks, dim=0)
        
        return blocks

    def _split_single_bdf_into_np_epochs(self, bdf):

        """
        Splits a BDF file into epochs based on annotations.
        
        Args:
            bdf: MNE Raw object with annotations
            
        Returns:
            numpy array of shape [n_epochs, n_channels, n_times]
        """
        if len(bdf.annotations) == 0:
            if self.verbose:
                print("No annotations found in BDF file, cannot create epochs")
            return np.array([])
        
        # Define epoch parameters
        #tmin = 1.5  # actual word start if we take it from the baseline
        tmin = 0 # we don't include the baseline
        tmax = 2   # actual word thinking length end
        
        # Create events from annotations
        annotations = bdf.annotations
        event_id = {str(desc): int(desc) for desc in annotations.description if desc.isdigit()}
        events, _ = mne.events_from_annotations(bdf, event_id=event_id, verbose=self.verbose)
        events = mne.pick_events(events, exclude=65536)
        #event_id = {code for code in events[:, 2]}
        
        # Create epochs
        epochs = mne.Epochs(
            bdf, 
            events, 
            event_id = 44, # start of word #13, # this is the event id for the word start
            tmin=tmin, 
            tmax=tmax, 
            baseline=(0, 0),
            preload=True,
            verbose=self.verbose
        )

        #word_start_idx = int(1.5 * self.default_rate)
        
        # Convert to numpy array
        epochs_data = epochs.get_data()

        """
        From the paper:
        At the beginning of each run, the condition was announced in
        the computer screen for a period of 3 seconds. In all cases, the order of the runs was: one pronounced speech,
        two inner speech and two visualized conditions. A one minute break between runs was given (inter-run break).
        """
        # so we are taking only inner speech idxs
        block_size = 5
        relative_inner_speech_indices = [1, 2]
        num_blocks = epochs_data.shape[0] // block_size
        
        all_inner_speech_indices = []
        for i in range(num_blocks):
            block_start_index = i * block_size
            for rel_idx in relative_inner_speech_indices:
                absolute_index = block_start_index + rel_idx
                all_inner_speech_indices.append(absolute_index)
        
        epochs_data = epochs_data[all_inner_speech_indices, :, 1:]


        #print(epochs_data.shape)
        #epochs_data = epochs_data[:, :, word_start_idx:]
        #print(epochs_data.shape)
        if self.verbose:
            print(f"Created {epochs_data.shape[0]} epochs from BDF file")
            
        return epochs_data
    
    def _split_bdfs_into_torch_epochs(self, eeg_data):
        """
        Splits the BDF files into numpy epochs.
        """
        epochs = []
        if self.verbose:
            progress_bar = tqdm(eeg_data, desc="Splitting bdfs into blocks")
        else:
            progress_bar = eeg_data

        for bdf in progress_bar:
            epochs.append(self._split_single_bdf_into_np_epochs(bdf))

        return torch.FloatTensor(np.concatenate(epochs, axis=0))
        
    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def get_chans(self, ch_names):
            chans = []
            for ch_name in ch_names:
                try:
                    chans.append(standard_1020.index(ch_name))
                except ValueError:
                    raise ValueError(f"Channel {ch_name} not found in standard_1020")
            return chans

    def __getitem__(self, index):
        #sample = pickle.load(open(self.files[index], "rb"))
        #data = sample["X"]
        #ch_names = sample["ch_names"]
        #data = torch.FloatTensor(data / 100)

        data = self.all_blocks[index]

        # splits the data into windows which are then indexed with the learnable embedding?
        time = data.size(1) // 200
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        #data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        data = data.view(data.size(0), -1, 200) # Reshape to [NumChannels, NumPatches, PatchLength]
        data = data.permute(1, 0, 2) # Reshape to [NumPatches, NumChannels, PatchLength]
        data = data.contiguous().view(-1, 200) # Reshape to [NumPatches * NumChannels, PatchLength]
        # this creates a data in a format 
        # [ch_1_patch_1, ch_2_patch_1, ch_3_patch_1, ..., ch_1_patch_2, ch_2_patch_2, ch_3_patch_2, ...]
        
        X = torch.zeros((self.block_size, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100))
            Y_raw = torch.zeros((self.block_size, 200))
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)
        
        # input_chans is the indices of the channels in the standard_1020 list
        # used for the spatial embedding
        input_chans = list(self.ch_names) * time
        input_chans.extend(['pad'] * (self.block_size - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        # input_time is the mask for padding zeros
        # ensure that padding zeros are not used in the attention mechanism
        input_time.extend([0] * (self.block_size - data.size(0)))
        input_time = torch.IntTensor(input_time)

        input_mask = torch.ones(self.block_size)
        input_mask[data.size(0):] = 0

        if self.GPT_training:
            # # gpt_mask is the mask for the GPT model
            # gpt_mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1, self.block_size, self.block_size)
            # num_chans = len(ch_names)
            # for i in range(time):
            #     gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1

            # gpt_mask[:, :, num_chans * time:] = 0
            # return X, input_chans, input_time, input_mask.bool(), gpt_mask.bool(), num_chans, data.size(0)
            warnings.warn("GPT training is not implemented yet")

        return X, Y_freq, Y_raw, input_chans, input_time, input_mask.bool()
    

class ZuCoLoader():
    """
    Notes:
    * the unit is microvolt

    Based partially on this:
    https://github.com/norahollenstein/zuco-benchmark
    And this:
    https://osf.io/r6ewq

    Here is the data:
    https://osf.io/q3zws/files/osfstorage# - I used Matlab files directories

    The following channels were removed from the ZuCo1 and ZuCo2 datasets:
    [E1, E8, E14, E17, E21, E25, E32, E48, E49, E56, E63, E68, E73, E81, E88, E94, E99, E107, E113, E119, E125, E126, E127, E128]
    
    The input shape is 105 because there is CZ reference electrode added - almost certainly the first channel is the refecence one - Cz.
    The Cz channel is non-zero, so it is most likely after the average reference was applied.

    """
    def __init__(self,
                 eeg_data: Union[str,
                    List[str],
                    ], 
                 block_size=1024,
                 extract_all_blocks=True, # if True, the fif files will be split into windows of a given length regardless of the events
                 window_length_s=1, # the length of the windows in seconds
                 processing_dict: Union[dict, str] = None,
                 GPT_training=False,
                 verbose=False):
        

        import warnings
        import re

        class MNEFilterWarning(Warning):
            pass

        # Custom warning filter function
        def filter_mne_warnings(message, category, filename, lineno, file=None, line=None):
            if category == RuntimeWarning and "filter length" in str(message):
                return 0  # Suppress the warning
            return -1  # Show other warnings

        # Register our custom filter
        warnings.filterwarnings("ignore", message=".*filter_length.*", category=RuntimeWarning)

        self.default_rate = 200
        self.block_size = block_size
        self.extract_all_blocks = extract_all_blocks
        self.window_length_s = window_length_s
        #self.processing_dict = processing_dict
        self.GPT_training = GPT_training
        self.verbose = verbose

        self.original_ch_names = [
            'CZ',
            'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20',
            'E22',
            'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39',
            'E40',
            'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58',
            'E59',
            'E60', 'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77',
            'E78',
            'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96',
            'E97',
            'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112',
            'E114',
            'E115', 'E116', 'E117', 'E118', 'E120', 'E121', 'E122', 'E123', 'E124']
        
        self.hydrocel_128_to_1005 = {
            'CZ': 'CZ',
            # Pre-frontal / Anterior Frontal
            'E22': 'Fp1',
            'E9': 'Fp2',
            'E16': 'Fpz',
            'E26': 'AF7',
            'E19': 'AF3',
            'E12': 'AFz',
            'E123': 'AF4',
            'E115': 'AF8',

            # Frontal
            'E27': 'F7',
            'E23': 'F5',
            'E24': 'F3',
            'E13': 'F1',
            'E11': 'Fz',
            'E118': 'F2',
            'E124': 'F4',
            'E114': 'F6',
            'E122': 'F8',

            # Fronto-Central / Fronto-Temporal
            'E33': 'FT7',
            'E28': 'FC5',
            'E31': 'FC3',
            'E20': 'FC1',
            'E6': 'FCz',
            'E109': 'FC2',
            'E116': 'FC4',
            'E111': 'FC6',
            'E117': 'FT8',
            
            # Temporal / Central
            'E45': 'T7',  # T7 often referred to as T3 in older systems
            'E37': 'C5',
            'E36': 'C3',
            'E47': 'C1',
            'E55': 'Cz',  # Assuming E55 is the vertex channel if E129 is VRef/ground
            'E80': 'C2',
            'E104': 'C4',
            'E105': 'C6',
            'E110': 'T8',  # T8 often referred to as T4 in older systems

            # Centro-Parietal / Temporo-Parietal
            'E40': 'TP7',
            'E41': 'CP5',
            'E42': 'CP3',
            'E53': 'CP1',
            'E79': 'CPz',
            'E87': 'CP2',
            'E93': 'CP4',
            'E103': 'CP6',
            'E97': 'TP8',

            # Parietal
            'E58': 'P7',  # P7 often referred to as T5 in older systems
            'E51': 'P5',
            'E52': 'P3',
            'E54': 'P1',
            'E62': 'Pz',
            'E90': 'P2',
            'E78': 'P4',
            'E96': 'P6',
            'E92': 'P8',  # P8 often referred to as T6 in older systems

            # Parieto-Occipital
            'E66': 'PO7',
            'E61': 'PO3',
            'E71': 'POz',
            'E85': 'PO4',
            'E86': 'PO8',

            # Occipital / Inion
            'E70': 'O1',
            'E75': 'Oz',
            'E83': 'O2',
            'E74': 'Iz', # Inion area

            # Inferior Row (Example - less consistently mapped than others)
            'E46': 'FT9', # Uncomment if needed, ensure uniqueness
            'E102': 'FT10',# Uncomment if needed, ensure uniqueness
            'E59': 'TP9', # Uncomment if needed, ensure uniqueness
            'E101': 'TP10',# Uncomment if needed, ensure uniqueness
            'E69': 'O9',  # Often near PO9 location # Uncomment if needed, ensure uniqueness
            'E95': 'O10', # Often near PO10 location # Uncomment if needed, ensure uniqueness
            'E60': 'P9',  # Uncomment if needed, ensure uniqueness
            'E98': 'P10'  # Uncomment if needed, ensure uniqueness
        }

        # maps hydrocel channels to 1020 channels
        # also takes the indices of the mapped hydrocel channels since not all of them are mapped and only part of the eeg channels are used
        self.not_used_hydrocel_128_channels = []
        self.used_hydrocel_ch_idxs = []
        self.ch_names = []
        for i, original_ch_name in enumerate(self.original_ch_names):
            if original_ch_name not in self.hydrocel_128_to_1005.keys():
                self.not_used_hydrocel_128_channels.append(original_ch_name)
            else:
                self.ch_names.append(self.hydrocel_128_to_1005[original_ch_name])
                self.used_hydrocel_ch_idxs.append(i)
        

        self.recording_paths = eeg_data
        if isinstance(self.recording_paths, str):
            self.recording_paths = [self.recording_paths]

        # assert that the eeg_data is a list of strings
        assert all(isinstance(item, str) for item in self.recording_paths), "eeg_data must be a list of strings"

        # processing the data during loading
        if self.verbose:
            data_loading_progress_bar = tqdm(self.recording_paths, desc="Loading data")
        else:
            data_loading_progress_bar = self.recording_paths
    
        self.eeg_data = []
        for mat_path in data_loading_progress_bar:
            self.eeg_data.append(self._load_mat_file(mat_path))

        # this is quite memory consuming, should be changed into appending self.eeg_data
        if self.extract_all_blocks:
            self.all_blocks = self._split_mat_recordings_into_all_blocks(self.eeg_data)
        else:
            # Not yet implemented - this will stop code execution
            raise NotImplementedError("Extracting single words is not yet implemented. Please set extract_all_blocks=True.")

        
    def _load_mat_file(self, mat_path):
        """
        Load mat files for the ZuCo data.
        """

        data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)['sentenceData']

        # returns list of sentences data - .rawData returns eeg data, .content returns the sentence and .word returns the sentence data - eeg and words etc.
        return data
    
    def _sentence_to_mne_raw(self, sentence):
        """
        Convert a sentence to an mne.io.Raw object.

        It filters and resample the data.
        """

        sentence_array = sentence.rawData
        sentence_array = sentence_array[self.used_hydrocel_ch_idxs, :]
        info = mne.create_info(ch_names=self.ch_names, sfreq=500, ch_types='eeg', verbose=self.verbose)

        mne_raw = mne.io.RawArray(sentence_array, info, verbose=self.verbose)
        
        # Suppress MNE RuntimeWarnings about filter_length - sentences are very short unfortunatelly
        with mne.utils.use_log_level('error'):
            mne_raw.notch_filter(50, verbose=self.verbose)

            mne_raw.filter(l_freq=1, h_freq=None, verbose=self.verbose)

            mne_raw.resample(200, verbose=self.verbose)

        return mne_raw


    def _split_mne_raw_sentence_into_all_blocks(self, mne_raw_sentence):
        """
        Split a sentence into all blocks.
        """

        eeg_array = mne_raw_sentence.get_data()
        block_length = self.window_length_s * self.default_rate

        blocks = []
        for i in range(0, eeg_array.shape[1], block_length):
            if i + block_length > eeg_array.shape[1]:
                if self.verbose:
                    print(f"Dropping last block of length {eeg_array.shape[1] - i} samples (shorter than {block_length} samples)")
                break
            blocks.append(eeg_array[:, i:i+block_length])

        blocks_np = np.array(blocks)
        return torch.FloatTensor(blocks_np)
    
    def _split_single_mat_recording_into_all_blocks(self, mat_recording):
        """
        Split a mat recording into all blocks.
        """

        all_blocks = []

        for sentence in mat_recording:
            try:
                mne_raw_sentence = self._sentence_to_mne_raw(sentence)
                blocks = self._split_mne_raw_sentence_into_all_blocks(mne_raw_sentence)
                all_blocks.append(blocks)
            except:
                if self.verbose:
                    print('Error in sentence:')
                    print(sentence)
                else:
                    pass

        #print('Length of all blocks for a single mat recording: ', len(all_blocks))
        return torch.concat(all_blocks, dim=0)
    

    def _split_mat_recordings_into_all_blocks(self, mat_recordings):
        """
        Split a list of mat recordings into all blocks.
        """

        all_blocks = []
        for mat_recording in mat_recordings:
            all_blocks.append(self._split_single_mat_recording_into_all_blocks(mat_recording))

        return torch.concat(all_blocks, dim=0)
    

    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def get_chans(self, ch_names):
            chans = []
            for ch_name in ch_names:
                try:
                    chans.append(standard_1020.index(ch_name))
                except ValueError:
                    raise ValueError(f"Channel {ch_name} not found in standard_1020")
            return chans
    
    def __getitem__(self, index):
        #sample = pickle.load(open(self.files[index], "rb"))
        #data = sample["X"]
        #ch_names = sample["ch_names"]
        #data = torch.FloatTensor(data / 100)

        data = self.all_blocks[index]

        # splits the data into windows which are then indexed with the learnable embedding?
        time = data.size(1) // 200
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        #data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        data = data.view(data.size(0), -1, 200) # Reshape to [NumChannels, NumPatches, PatchLength]
        data = data.permute(1, 0, 2) # Reshape to [NumPatches, NumChannels, PatchLength]
        data = data.contiguous().view(-1, 200) # Reshape to [NumPatches * NumChannels, PatchLength]
        # this creates a data in a format 
        # [ch_1_patch_1, ch_2_patch_1, ch_3_patch_1, ..., ch_1_patch_2, ch_2_patch_2, ch_3_patch_2, ...]
        
        X = torch.zeros((self.block_size, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100))
            Y_raw = torch.zeros((self.block_size, 200))
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)
        
        # input_chans is the indices of the channels in the standard_1020 list
        # used for the spatial embedding
        input_chans = list(self.ch_names) * time
        input_chans.extend(['pad'] * (self.block_size - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        
        # input_time is the mask for padding zeros
        # ensure that padding zeros are not used in the attention mechanism
        input_time.extend([0] * (self.block_size - data.size(0)))
        input_time = torch.IntTensor(input_time)

        input_mask = torch.ones(self.block_size)
        input_mask[data.size(0):] = 0

        if self.GPT_training:
            # # gpt_mask is the mask for the GPT model
            # gpt_mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1, self.block_size, self.block_size)
            # num_chans = len(ch_names)
            # for i in range(time):
            #     gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1

            # gpt_mask[:, :, num_chans * time:] = 0
            # return X, input_chans, input_time, input_mask.bool(), gpt_mask.bool(), num_chans, data.size(0)
            warnings.warn("GPT training is not implemented yet")

        return X, Y_freq, Y_raw, input_chans, input_time, input_mask.bool()
    

