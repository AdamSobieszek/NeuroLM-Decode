"""
by Wei-Bang Jiang (modified)
https://github.com/935963004/NeuroLM
"""

from torch.utils.data import Dataset
import torch
from einops import rearrange
import pickle
import numpy as np

def reshape_data_by_channels(data, num_channels=23):
    if len(data.shape)==2:
        data=data.unsqueeze(0)
    batch_size, num_tokens, time_samples = data.shape
    
    # Calculate how many tokens per channel we have
    tokens_per_channel = num_tokens // num_channels
    
    # Create output tensor
    reshaped_data = torch.zeros((batch_size, num_channels, tokens_per_channel * time_samples))
    
    reshaped_data = data[:,:tokens_per_channel*num_channels].reshape(batch_size, num_channels, tokens_per_channel * time_samples)
    return reshaped_data

def inverse_reshape_data_by_channels(reshaped_data, time_samples=200, num_channels=23):
    batch_size, _, total_length = reshaped_data.shape
    tokens_per_channel = total_length // time_samples
    num_tokens = tokens_per_channel * num_channels
    
    # Create output tensor
    original_data = torch.zeros((batch_size, num_tokens, time_samples))
    original_data = reshaped_data.reshape(batch_size, num_tokens, time_samples)
    
    return original_data

def std_norm(x):
    mean = torch.mean(x, dim=(0, 1), keepdim=True)
    std = torch.std(x, dim=(0, 1), keepdim=True)
    x = (x - mean) / std
    return x

def norm_whole_channels(data, num_channels=23):
    if len(data.shape)==2:
        data=data.unsqueeze(0)
    batch_size, num_tokens, time_samples = data.shape
    reshaped_data = reshape_data_by_channels(data, num_channels)
    reshaped_data = std_norm(reshaped_data)
    reshaped_data = inverse_reshape_data_by_channels(reshaped_data)

    return reshaped_data 
     
# Create a Hann window function for FFT
def create_hann_window(window_size):
    return 0.5 * (1 - torch.cos(2 * np.pi * torch.arange(window_size) / (window_size - 1)))

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
smaller_ch_names_list = ['CZ',
 'O1',
 'O2',
 'P3',
 'C4',
 'P4',
 'FP2',
 'FZ',
 'FP1',
 'F4',
 'F8',
 'C3',
 'F3',
 'F7']

# Define EEG frequency bands
eeg_bands = {
    'delta': (0.5, 4),    # Delta waves (deep sleep)
    'theta': (4, 8),      # Theta waves (drowsiness)
    'alpha': (8, 13),     # Alpha waves (relaxed wakefulness)
    'beta': (13, 30),     # Beta waves (active thinking)
    'gamma': (30, 100)    # Gamma waves (cognitive processing)
}

class PickleLoader(Dataset):
    def __init__(self, files, block_size=1024, sampling_rate=200, GPT_training=False, window_overlap=150):
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.GPT_training = GPT_training
        self.window_overlap = window_overlap
        self.window_size = 200  # Default window size
        self.hann_window = create_hann_window(self.window_size*3)

    def __len__(self):
        return len(self.files)
    
    def std_norm(self, x, dim=(0,1)):
        eps = 1e-10
        mean = torch.mean(x, dim=dim, keepdim=True)
        std = torch.std(x, dim=dim, keepdim=True)
        return (x - mean) / (std + eps)

    def get_chans(self, ch_names):
        return [standard_1020.index(ch) for ch in ch_names]

    def compute_windowed_fft(self, data, window_size=200, overlap=100):
        num_channels, time_samples = data.shape
        num_windows = (time_samples - window_size) // window_size + 1
        freq_data = []
        for ch in range(num_channels):
            ch_freq = []
            for w in range(num_windows):
                start_idx = max([0,w * window_size - overlap])
                end_idx = start_idx + window_size + 2*overlap
                segment = data[ch, start_idx:end_idx] * self.hann_window
                fft_result = torch.fft.rfft(segment)
                amplitude = torch.log(torch.abs(fft_result) + 1e-10)
                ch_freq.append(amplitude)
            freq_data.append(torch.stack(ch_freq).mean(dim=0))
        return torch.stack(freq_data)

    def extract_extended_window(self, data, num_channels, time_index):
        total_samples = data.size(1)
        center_idx = time_index * 200
        start_idx = max(0, center_idx - 200)
        end_idx = min(total_samples, center_idx + 400)
        extended = data[:, start_idx:end_idx]
        if extended.size(1) < 600:
            pad_size = 600 - extended.size(1)
            pad = torch.zeros((num_channels, pad_size))
            extended = torch.cat((pad, extended), dim=1) if start_idx == 0 else torch.cat((extended, pad), dim=1)
        return extended

    def __getitem__(self, index):
        path = self.files[index]
        # 1) load
        with open(path, 'rb') as f:
            sample = pickle.load(f)

        # 2) if we've already cached the preprocessed tensors and we're not in GPT mode, just return them
        if not self.GPT_training and 'Y_freq' in sample:
            return (sample['X'], sample['Y_freq'], sample['Y_raw'],
                    sample['input_chans'], sample['input_time'], sample['input_mask'])

        # 3) otherwise we must compute them from raw
        raw_X   = sample['X']        # shape (N_chan, T)
        ch_names= sample['ch_names']

        data = torch.FloatTensor(raw_X / 100.0)
        data = self.std_norm(data, (0,1))
        time = data.size(1) // 200
        num_channels = len(ch_names)
        original_data = data.clone()

        # build X
        data_windows = rearrange(data, 'N (A T) -> (A N) T', T=200)
        X = torch.zeros((self.block_size, 200), dtype=torch.float32)
        X[:data_windows.size(0)] = data_windows

        # build Y_freq, Y_raw
        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100), dtype=torch.float32)
            
            Y_raw  = X
            for t in range(time):
                start_idx = t * num_channels
                end_idx   = start_idx + num_channels
                ext_win = self.extract_extended_window(original_data, num_channels, t)* self.hann_window
                fft_result = torch.fft.rfft(ext_win)
                freq_data = torch.log(torch.abs(fft_result) + 1e-10)
                freq_data = freq_data[:, :100]
                scaling = torch.arange(100, 500, 4, dtype=torch.float32).unsqueeze(0)
                freq_data = freq_data / scaling * 100.0/3
                for ch_idx in range(num_channels):
                    Y_freq[start_idx+ch_idx] = freq_data[ch_idx]

        # build channel and time indices and mask
        input_time = [i for i in range(time) for _ in range(num_channels)]
        input_time.extend([0] * (self.block_size - data_windows.size(0)))
        input_time = torch.IntTensor(input_time)

        input_chans = (list(ch_names) * time) + ['pad'] * (self.block_size - data_windows.size(0))
        input_chans = torch.IntTensor(self.get_chans(input_chans))

        input_mask = torch.ones(self.block_size, dtype=torch.bool)
        input_mask[data_windows.size(0):] = False

        # if GPT mode, do original GPT branch (no caching)
        if self.GPT_training:
            gpt_mask = torch.tril(torch.ones(self.block_size, self.block_size, dtype=torch.bool))
            gpt_mask = gpt_mask.view(1, self.block_size, self.block_size)
            for i in range(time):
                i0 = i * num_channels
                i1 = i0 + num_channels
                gpt_mask[:, i0:i1, i0:i1] = True
            gpt_mask[:, :, num_channels * time:] = False
            return X, input_chans, input_time, input_mask, gpt_mask, num_channels, data_windows.size(0)

        # 4) cache the freshly computed tensors by overwriting the .pkl
        to_cache = {
            'X': X,
            'Y_freq': Y_freq,
            'Y_raw':  Y_raw,
            'input_chans': input_chans,
            'input_time':  input_time,
            'input_mask':  input_mask
        }
        with open(path, 'wb') as f:
            pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 5) return
        return X, Y_freq, Y_raw, input_chans, input_time, input_mask

import random
import torch
from torch.utils.data import Dataset
class IvoLoader(PickleLoader):
    """
    50% of the time returns a random row from a single big in‐RAM tensor of shape [M, N_tokens, 200],
    50% of the time falls back to your old PickleLoader on-disk cache.
    Always returns (X, Y_freq, Y_raw, input_chans, input_time, input_mask).
    """
    def __init__(self,
                 pickle_files,       # list of your already‐cached .pkl files
                 block_size=1024,
                 window_size=200,
                 freq_bins=100,
                 window_overlap=100,
                 mem_tensor_path = "/Users/adamsobieszek/PycharmProjects/InnerSpeechMVPv1/ivo_pretraining",    # path to your [600,460,200].pt
                ):
        super().__init__(pickle_files)
        # 1) load the one big tensor
        self.mem_data = torch.load(mem_tensor_path)   # Tensor[M, N_tokens, 200]
        assert self.mem_data.ndim == 3
        self.M, self.N_tokens, self.T = self.mem_data.shape
        assert self.T == window_size

        # 2) hard‐coded membrane channel names (46 of them)
        self.mem_ch_names = [
            'FP1','FPZ','FP2','AF7','AF3','AFZ','AF4','AF8',
            'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FC2','FC4','FC6',
            'T7','C5','C3','C1','CZ','C2','C4','C6',
            'CP5','CP3','CP1','CP2','CP4','CP6',
            'P7','P5','P3','P1','P2','P4','O1','O2'
        ]
        self.num_channels = len(self.mem_ch_names)
        assert self.N_tokens % self.num_channels == 0, "N_tokens must be divisible by num_channels"
        self.time_steps = self.N_tokens // self.num_channels

        # 3) copy your old PickleLoader for the on‐disk side
        self.file_ds = PickleLoader(
            files=pickle_files,
            
            GPT_training=False,
            window_overlap=window_overlap
        )
        self.N_files = len(self.file_ds)

        # 4) shared params
        self.block_size    = block_size
        self.window_size   = window_size
        self.freq_bins     = freq_bins
        self.window_overlap= window_overlap
        self.hann_window   = create_hann_window(window_size*3)   # [200]
        # build your physiological scaling factor once
        self.scaling       = torch.arange(100, 500, 4, dtype=torch.float32).unsqueeze(0)  # [1,100]

    def __len__(self):
        # exactly half in‐RAM draws, half on‐disk draws per epoch
        return 2 * self.N_files

    def get_chans(self, ch_names):
        # same as your original
        return [standard_1020.index(ch) for ch in ch_names]

    def __getitem__(self, idx):
        # even idx → in‐RAM
        if idx % 2 == 0:
            # pick a random row
            r = random.randrange(self.M)
            row = self.mem_data[r]          # [N_tokens, 200]
            n = len(row)
            num_channels = 46
            # 1) X
            
            original_data = reshape_data_by_channels(row, 46).squeeze()
            original_data = self.std_norm(original_data, (0,1))
            row = rearrange(original_data, 'N (A T) -> (A N) T', T=200)

            Y_freq = torch.zeros((self.block_size, 100), dtype=torch.float32)
            time = original_data.size(1) // 200
            for t in range(time):

                start_idx = t * num_channels
                end_idx   = start_idx + num_channels
                ext_win = self.extract_extended_window(original_data, 46, t)* self.hann_window
                fft_result = torch.fft.rfft(ext_win)
                freq_data = torch.log(torch.abs(fft_result) + 1e-10)
                # Pool amplitudes in FFT output to reduce from 300 to 100 frequency bins
                freq_data = freq_data[:, :300]  # Ensure we're working with the first 300 bins
                freq_data = freq_data.reshape(freq_data.size(0), 100, 3).mean(dim=-1)
                scaling = torch.arange(100, 500, 4, dtype=torch.float32).unsqueeze(0)
                freq_data = freq_data / scaling * 100.0/3
                for ch_idx in range(num_channels):
                    Y_freq[start_idx+ch_idx] = freq_data[ch_idx]

            X = torch.zeros(self.block_size, self.window_size, dtype=row.dtype)
            X[:n] = row

            # 2) Y_raw = identical to X
            Y_raw = X.clone()

           
            # 4) input_chans
            #    repeat mem_ch_names once per time step
            chan_list = self.mem_ch_names * self.time_steps
            chan_list += ['pad'] * (self.block_size - n)
            input_chans = torch.IntTensor(self.get_chans(chan_list))

            # 5) input_time
            time_list = [t for t in range(self.time_steps) for _ in range(self.num_channels)]
            time_list += [0] * (self.block_size - n)
            input_time = torch.IntTensor(time_list)

            # 6) input_mask
            input_mask = torch.ones(self.block_size, dtype=torch.bool)
            input_mask[n:] = False

            return X, Y_freq, Y_raw, input_chans, input_time, input_mask
            

        # odd idx → on‐disk
        else:
            file_idx = idx // 2
            
            path = self.files[file_idx]
            # 1) load
            with open(path, 'rb') as f:
                sample = pickle.load(f)

            Y_freq = torch.zeros((self.block_size, 100), dtype=torch.float32)
            Y_raw  = sample['X']      
            num_channels = 23
            original_data = reshape_data_by_channels(sample['X'][:(8800*23)//200]).squeeze()[:,:8800]

            time = original_data.size(1) // 200
            for t in range(time):
                start_idx = t * num_channels
                end_idx   = start_idx + num_channels
                ext_win = self.extract_extended_window(original_data, num_channels, t) * self.hann_window
                fft_result = torch.fft.rfft(ext_win)
                freq_data = torch.log(torch.abs(fft_result) + 1e-10)
                freq_data = freq_data[:, :100]
                scaling = torch.arange(100, 500, 4, dtype=torch.float32).unsqueeze(0)
                freq_data = freq_data / scaling * 100.0/3
                for ch_idx in range(num_channels):
                    Y_freq[start_idx+ch_idx] = freq_data[ch_idx]
            
            return (sample['X'], Y_freq, Y_raw,
                    sample['input_chans'], sample['input_time'], sample['input_mask'])
        

class InnerSpeechLoader(PickleLoader):
    """
    50% of the time returns a random row from a single big in‐RAM tensor of shape [M, N_tokens, 200],
    50% of the time falls back to your old PickleLoader on-disk cache.
    Always returns (X, Y_freq, Y_raw, input_chans, input_time, input_mask).
    """
    def __init__(self,       # list of your already‐cached .pkl files
                 block_size=1024,
                 window_size=200,
                 freq_bins=100,
                 window_overlap=100,
                 mem_tensor_path = "/Users/adamsobieszek/PycharmProjects/InnerSpeechMVPv1/ivo_pretraining",    # path to your [600,460,200].pt
                ):
        # 1) load the one big tensor
        self.mem_data = torch.load(mem_tensor_path)   # Tensor[M, N_tokens, 200]
        assert self.mem_data.ndim == 3
        self.M, self.N_tokens, self.T = self.mem_data.shape
        assert self.T == window_size

        # 2) hard‐coded membrane channel names (46 of them)
        self.mem_ch_names = [
            'FP1','FPZ','FP2','AF7','AF3','AFZ','AF4','AF8',
            'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
            'FT7','FC5','FC3','FC1','FC2','FC4','FC6',
            'T7','C5','C3','C1','CZ','C2','C4','C6',
            'CP5','CP3','CP1','CP2','CP4','CP6',
            'P7','P5','P3','P1','P2','P4','O1','O2'
        ]
        self.num_channels = len(self.mem_ch_names)
        assert self.N_tokens % self.num_channels == 0, "N_tokens must be divisible by num_channels"
        self.time_steps = self.N_tokens // self.num_channels


        # 4) shared params
        self.block_size    = block_size
        self.window_size   = window_size
        self.freq_bins     = freq_bins
        self.window_overlap= window_overlap
        self.hann_window   = create_hann_window(window_size*3)   # [200]
        # build your physiological scaling factor once
        self.scaling       = torch.arange(100, 500, 4, dtype=torch.float32).unsqueeze(0)  # [1,100]

    def __len__(self):
        # exactly half in‐RAM draws, half on‐disk draws per epoch
        return len(self.mem_data)

    def get_chans(self, ch_names):
        # same as your original
        return [standard_1020.index(ch) for ch in ch_names]

    def __getitem__(self, idx):
            # pick a random row
            r = random.randrange(self.M)
            row = self.mem_data[r]          # [N_tokens, 200]
            n = len(row)
            num_channels = 46
            # 1) X
            
            original_data = reshape_data_by_channels(row, 46).squeeze()
            original_data = self.std_norm(original_data, (0,1))
            row = rearrange(original_data, 'N (A T) -> (A N) T', T=200)

            Y_freq = torch.zeros((self.block_size, 100), dtype=torch.float32)
            time = original_data.size(1) // 200
            for t in range(time):

                start_idx = t * num_channels
                end_idx   = start_idx + num_channels
                ext_win = self.extract_extended_window(original_data, 46, t)* self.hann_window
                fft_result = torch.fft.rfft(ext_win)
                freq_data = torch.log(torch.abs(fft_result) + 1e-10)
                # Pool amplitudes in FFT output to reduce from 300 to 100 frequency bins
                freq_data = freq_data[:, :300]  # Ensure we're working with the first 300 bins
                freq_data = freq_data.reshape(freq_data.size(0), 100, 3).mean(dim=-1)
                scaling = torch.arange(100, 500, 4, dtype=torch.float32).unsqueeze(0)
                freq_data = freq_data / scaling * 100.0/3
                for ch_idx in range(num_channels):
                    Y_freq[start_idx+ch_idx] = freq_data[ch_idx]

            X = torch.zeros(self.block_size, self.window_size, dtype=row.dtype)
            X[:n] = row

            # 2) Y_raw = identical to X
            Y_raw = X.clone()

           
            # 4) input_chans
            #    repeat mem_ch_names once per time step
            chan_list = self.mem_ch_names * self.time_steps
            chan_list += ['pad'] * (self.block_size - n)
            input_chans = torch.IntTensor(self.get_chans(chan_list))

            # 5) input_time
            time_list = [t for t in range(self.time_steps) for _ in range(self.num_channels)]
            time_list += [0] * (self.block_size - n)
            input_time = torch.IntTensor(time_list)

            # 6) input_mask
            input_mask = torch.ones(self.block_size, dtype=torch.bool)
            input_mask[n:] = False

            return X, Y_freq, Y_raw, input_chans, input_time, input_mask
            

