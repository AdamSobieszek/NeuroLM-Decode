# File: train_lblm_mstp.py

import os
import time
import argparse
import signal
import sys
from contextlib import nullcontext
import glob # For finding files

import math
import numpy as np
import torch
import torch.nn.functional as F # For potential use, though model handles loss
# import torch._dynamo.config # Not strictly needed unless you're tweaking dynamo
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import collections
import io
import matplotlib.pyplot as plt

# Assuming 'model2' package is in PYTHONPATH or in the same directory structure
try:
    from model2 import InnerSpeech_LBLM_MSTP
    from lblm_dataset import EEGPickleDataset # Assuming you saved it as eeg_pickle_dataset.py
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'model2' package and 'eeg_pickle_dataset.py' are correctly set up.")
    sys.exit(1)

from pathlib import Path

# --- Global DDP Variables (as in your script) ---
master_process = True # Default for non-DDP
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = torch.float32 # Default, will be set in init
ctx = nullcontext()
ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
device_type = 'cuda' if 'cuda' in device else 'cpu'
ddp = False # Will be set in init
running = True # For signal handler

import wandb # Add this import
import matplotlib.pyplot as plt
import io # For saving plot to a buffer

# ... (other imports) ...

import matplotlib.pyplot as plt
import io
import random # For selecting block start

# ... (other imports) ...

def log_reconstruction_plot_to_wandb(model, val_batch, epoch, global_step, device, 
                                     ptdtype_val, ctx_val, args_cli_val, 
                                     plot_masking_strategy='random', # 'random', 'block'
                                     block_mask_size=5): 
    """
    Generates and logs a plot of EEG reconstructions for a sample.
    plot_masking_strategy: 'random' (uses model's internal masking) or 
                           'block' (masks a contiguous block for plotting).
    """
    if not args_cli_val.wandb_log or not master_process:
        return

    current_model_training_state = model.training
    model.eval()

    x_eeg_val = val_batch['x_raw_eeg'].to(device, non_blocking=True)
    subject_ids_val = val_batch['subject_id'].to(device, non_blocking=True)
    if ptdtype_val != torch.float32:
        x_eeg_val = x_eeg_val.to(ptdtype_val)

    sample_idx_in_batch = 0
    # For block masking, let's process all channels of the sample together to see context
    # single_x_raw_eeg will be [1, NumChannels, Timepoints]
    single_x_raw_eeg = x_eeg_val[sample_idx_in_batch:sample_idx_in_batch+1] 
    single_subject_id = subject_ids_val[sample_idx_in_batch:sample_idx_in_batch+1] if subject_ids_val is not None else None

    with torch.no_grad():
        # Get initial processing from the model's input_processor
        # This gives us the patches and the original random mask (if mask_ratio > 0)
        # embedded_patches: [NumChannels, NumPatches, embed_dim] for this single batch item
        # original_patches_for_target: [1, NumChannels, NumPatches, PatchLength]
        # masked_indices_bool_internal: [NumChannels, NumPatches] (after reshaping from B*M)
        
        _model_ref = model.module if ddp else model
        
        # Run input_processor on the single sample (all its channels)
        # embedded_patches_all_ch: [1*NumChannels, NumPatches, embed_dim]
        # original_patches_all_ch: [1, NumChannels, NumPatches, PatchLength]
        # masked_indices_bool_internal_flat: [1*NumChannels, NumPatches]
        embedded_patches_all_ch, original_patches_all_ch, masked_indices_bool_internal_flat, rev_in_mean, rev_in_std = \
            _model_ref.input_processor(single_x_raw_eeg, single_subject_id)

        num_actual_channels = single_x_raw_eeg.shape[1]
        num_patches_seq = embedded_patches_all_ch.shape[1]

        # This is the mask that input_processor would have generated based on its mask_ratio
        # masked_indices_bool_internal = masked_indices_bool_internal_flat.reshape(num_actual_channels, num_patches_seq)


        # --- Define the actual mask to use for this plot ---
        if plot_masking_strategy == 'block':
            # Create a block mask for one specific channel
            channel_idx_to_plot = 0 # Plotting reconstructions for this channel
            plot_masked_indices_bool_flat = torch.zeros_like(masked_indices_bool_internal_flat, dtype=torch.bool)
            
            if num_patches_seq > block_mask_size:
                start_idx = random.randint(0, num_patches_seq - block_mask_size -1) # Ensure block fits
                # Mask a block only for the channel_idx_to_plot
                # The input to conformer is [B*M, NumPatches, ...]. Here B=1.
                # So, index for the chosen channel is just channel_idx_to_plot.
                plot_masked_indices_bool_flat[channel_idx_to_plot, start_idx : start_idx + block_mask_size] = True
            else:
                # If not enough patches for a full block, mask all for that channel
                print(f"W&B Plot: Not enough patches ({num_patches_seq}) for block mask size ({block_mask_size}). Masking all for channel {channel_idx_to_plot}.")
                plot_masked_indices_bool_flat[channel_idx_to_plot, :] = True
            
            final_masked_indices_for_conformer_input = plot_masked_indices_bool_flat

        elif plot_masking_strategy == 'random':
            # Use the mask generated by input_processor (which depends on its internal mask_ratio)
            final_masked_indices_for_conformer_input = masked_indices_bool_internal_flat
            channel_idx_to_plot = 0 # Still pick one channel to visualize from the random mask
        else:
            raise ValueError(f"Unknown plot_masking_strategy: {plot_masking_strategy}")

        conformer_input = embedded_patches_all_ch.clone()
        conformer_input[final_masked_indices_for_conformer_input] = 0.0 # Zero out based on the chosen strategy

        backbone_output = _model_ref.conformer_backbone(conformer_input, attention_mask=final_masked_indices_for_conformer_input)
        
        # Select outputs for the *specific channel we want to plot* and *its masked patches*
        # final_masked_indices_for_conformer_input is [NumChannels, NumPatches] for this single sample
        # backbone_output is [NumChannels, NumPatches, d_model]
        
        # Get the mask for the specific channel we are plotting
        masked_patches_for_selected_channel_bool = final_masked_indices_for_conformer_input[channel_idx_to_plot]
        
        # Get backbone outputs only for the channel_idx_to_plot
        backbone_output_selected_channel = backbone_output[channel_idx_to_plot] # [NumPatches, d_model]
        
        # Select the outputs that were actually masked for this channel
        masked_backbone_outputs_selected_channel = backbone_output_selected_channel[masked_patches_for_selected_channel_bool]
        
        if masked_backbone_outputs_selected_channel.shape[0] == 0:
            print(f"W&B Plot ({plot_masking_strategy}): No patches were masked for selected channel {channel_idx_to_plot}. Skipping plot.")
            if current_model_training_state: model.train()
            return

        pred_wave_masked, pred_amplitude_masked, pred_phase_masked = \
            _model_ref.prediction_heads(masked_backbone_outputs_selected_channel)

        # Prepare Targets for the *selected channel's masked patches*
        # original_patches_all_ch: [1, NumChannels, NumPatches, PatchLength]
        original_patches_selected_channel = original_patches_all_ch[0, channel_idx_to_plot] # [NumPatches, PatchLength]
        target_wave_patches_masked = original_patches_selected_channel[masked_patches_for_selected_channel_bool]

        if _model_ref.input_processor.use_rev_in and rev_in_mean is not None and rev_in_std is not None:
            # rev_in_mean/std are [1, NumChannels, NumPatches, 1]
            rev_in_mean_selected_channel = rev_in_mean[0, channel_idx_to_plot] # [NumPatches, 1]
            rev_in_std_selected_channel = rev_in_std[0, channel_idx_to_plot]   # [NumPatches, 1]
            
            masked_means = rev_in_mean_selected_channel[masked_patches_for_selected_channel_bool]
            masked_stds = rev_in_std_selected_channel[masked_patches_for_selected_channel_bool]
            
            pred_wave_masked = _model_ref.input_processor.rev_in_denorm(pred_wave_masked, masked_means, masked_stds)

        target_amplitude_masked, target_phase_masked = \
            _model_ref._calculate_fft_targets(target_wave_patches_masked)

    # --- Data for Plotting ---
    num_patches_to_plot = min(5, pred_wave_masked.shape[0]) # Plot up to 5 of the masked patches
    if num_patches_to_plot == 0:
        print(f"W&B Plot ({plot_masking_strategy}): Not enough effectively masked patches on channel {channel_idx_to_plot} to plot.")
        if current_model_training_state: model.train()
        return

    # Take the first num_patches_to_plot from the (potentially many) masked ones
    pred_w = pred_wave_masked[:num_patches_to_plot].detach().cpu().float().numpy()
    target_w = target_wave_patches_masked[:num_patches_to_plot].detach().cpu().float().numpy()
    pred_a = pred_amplitude_masked[:num_patches_to_plot].detach().cpu().float().numpy()
    target_a = target_amplitude_masked[:num_patches_to_plot].detach().cpu().float().numpy()
    pred_p = pred_phase_masked[:num_patches_to_plot].detach().cpu().float().numpy()
    target_p = target_phase_masked[:num_patches_to_plot].detach().cpu().float().numpy()

    patch_len = args_cli_val.patch_length
    fft_len = target_a.shape[-1]

    fig, axs = plt.subplots(3, num_patches_to_plot, figsize=(3.5 * num_patches_to_plot, 8), sharey='row', squeeze=False)
    # Squeeze=False ensures axs is always 2D

    for i in range(num_patches_to_plot):
        axs[0, i].plot(np.arange(patch_len), target_w[i], label='Target', color='blue', linestyle='-')
        axs[0, i].plot(np.arange(patch_len), pred_w[i], label='Pred.', color='red', linestyle='--')
        axs[0, i].set_title(f'Wave (Patch {i+1})')
        if i == 0: axs[0, i].set_ylabel('Amplitude')
        axs[0, i].legend(fontsize='x-small')

        axs[1, i].plot(np.arange(fft_len), target_a[i], label='Target', color='blue', linestyle='-')
        axs[1, i].plot(np.arange(fft_len), pred_a[i], label='Pred.', color='red', linestyle='--')
        axs[1, i].set_title(f'FFT Amp. (Patch {i+1})')
        if i == 0: axs[1, i].set_ylabel('Magnitude')
        axs[1, i].legend(fontsize='x-small')

        axs[2, i].plot(np.arange(fft_len), target_p[i], label='Target', color='blue', linestyle='-')
        axs[2, i].plot(np.arange(fft_len), pred_p[i], label='Pred.', color='red', linestyle='--')
        axs[2, i].set_title(f'FFT Phase (Patch {i+1})')
        if i == 0: axs[2, i].set_ylabel('Radians')
        axs[2, i].set_xlabel('Time/Freq Bin')
        axs[2, i].legend(fontsize='x-small')

    plt.suptitle(f'Epoch {epoch} - Recon. (Ch {channel_idx_to_plot}, Sample {sample_idx_in_batch}, Strategy: {plot_masking_strategy})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    wandb.log({f"Val Recon/{plot_masking_strategy} Strategy": wandb.Image(plt)}, step=global_step)
    plt.close(fig)
    
    if current_model_training_state: model.train() # Restore original training state
def log_lamb_optimizer_stats_to_wandb(optimizer, global_step: int):
    """Log LAMB optimizer statistics (norms, trust ratio) to W&B."""
    global args_cli
    if not args_cli.wandb_log or not master_process: # Use global args from get_cli_args()
        return

    lamb_stats = collections.defaultdict(list)
    for group_idx, group in enumerate(optimizer.param_groups):
        for p_idx, p in enumerate(group['params']):
            if p.grad is None: # Skip params without grads (e.g., frozen)
                continue
            state = optimizer.state[p]
            if state: # Ensure state exists for this parameter
                # param_name = f"group{group_idx}_param{p_idx}" # Could try to get actual param names if needed
                if 'weight_norm' in state:
                    lamb_stats['lamb/weight_norm'].append(state['weight_norm'])
                if 'adam_norm' in state:
                    lamb_stats['lamb/adam_norm'].append(state['adam_norm'])
                if 'trust_ratio' in state:
                    lamb_stats['lamb/trust_ratio'].append(state['trust_ratio'])

    log_data = {}
    for k, v_list in lamb_stats.items():
        if v_list:
            # Log as histogram
            wandb.log({f"{k}_hist": wandb.Histogram(v_list)}, step=global_step)
            # Log mean/min/max as scalars
            v_tensor = torch.tensor(v_list)
            log_data[f"{k}_mean"] = v_tensor.mean().item()
            log_data[f"{k}_min"] = v_tensor.min().item()
            log_data[f"{k}_max"] = v_tensor.max().item()

    if log_data:
        wandb.log(log_data, step=global_step)
# --- Signal Handler ---
def signal_handler_interrupt(sig, frame):
    global running
    print('Keyboard interrupt received. Attempting to save checkpoint and exit...')
    running = False

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(math.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule

def init_ddp(args_ddp_backend):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, ptdtype, device_type, ddp_local_rank, args_cli
    
    ddp_flag = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp_flag:
        init_process_group(backend=args_ddp_backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        ddp = True
        print(f"DDP initialized: Rank {ddp_rank}/{ddp_world_size}, Local Rank {ddp_local_rank}, Device {device}")
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        # device is already 'cuda' or 'cpu'
        ddp = False
        print(f"Running in non-DDP mode on device: {device}")

    torch.manual_seed(args_cli.seed + seed_offset)
    if device == 'cuda':
        torch.cuda.manual_seed(args_cli.seed + seed_offset) # For current GPU

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Global ptdtype (PyTorch dtype)
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args_cli.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    print(f"Using dtype: {args_cli.dtype} ({ptdtype}), Context: {type(ctx)}")


def get_model_args(args_cli):
    """Creates a dictionary of model parameters from CLI args."""
    return {
        "patch_length": args_cli.patch_length,
        "patch_stride": args_cli.patch_stride,
        "num_subjects": args_cli.num_subjects,
        "subject_embed_dim": args_cli.subject_embed_dim,
        "mask_ratio": args_cli.mask_ratio,
        "use_rev_in": not args_cli.no_rev_in, # Invert logic for store_true
        "eeg_channels": args_cli.eeg_channels,
        "conformer_input_dim": args_cli.conformer_embed_dim, # Renamed for clarity
        "conformer_num_layers": args_cli.conformer_layers,
        "conformer_d_model": args_cli.conformer_d_model,
        "conformer_n_head": args_cli.conformer_n_head,
        "conformer_d_ff": args_cli.conformer_d_ff,
        "conformer_dropout_rate": args_cli.dropout,
        "conformer_conv_kernel_size": args_cli.conformer_conv_kernel,
        "huber_delta": args_cli.huber_delta,
        "lambda_amplitude": args_cli.lambda_amplitude,
        "lambda_phase": args_cli.lambda_phase
    }

def main_train_loop():
    global running, master_process, device, ptdtype, ctx, ddp, ddp_world_size, ddp_rank, args_cli

    init_ddp(args_cli.ddp_backend) # Initialize DDP and device settings


    if master_process:
        os.makedirs(args_cli.out_dir, exist_ok=True)
        if args_cli.wandb_log:
            try:
                wandb.init(project=args_cli.wandb_project,name=args_cli.wandb_run_name or args_cli.out_dir, config=args_cli) #args_cli.wandb_run_name
                print("Weights & Biases initialized.")
            except Exception as e:
                print(f"Error initializing Weights & Biases: {e}. Proceeding without W&B.")
                args_cli.wandb_log = False # Disable if init fails

        # Could also init wandb here if args_cli.wandb_log and master_process

    # --- Dataloader Setup ---
    print('Preparing dataloaders...')
    # Using glob for simplicity, Path.rglob is also good.
    train_pickle_files = glob.glob(os.path.join(args_cli.dataset_dir, "*.pkl"))

    # Remove any corrupt/unreadable pickle files
    import pickle
    bad_files = []
    for f in list(train_pickle_files):  # Use a copy since we may remove from the list
        try:
            with open(f, 'rb') as pf:
                pickle.load(pf)
        except Exception as e:
            print(f"Corrupt or unreadable pickle file detected and will be removed: {f} ({e})")
            try:
                os.remove(f)
            except Exception as remove_e:
                print(f"Failed to remove file {f}: {remove_e}")
            bad_files.append(f)
    # Remove bad files from the list
    train_pickle_files = [f for f in train_pickle_files if f not in bad_files]


    
    # Example: Splitting train_pickle_files into train and val
    # This is a simple split, consider more robust methods for real datasets
    if not train_pickle_files:
        print(f"Error: No pickle files found in {args_cli.dataset_dir}. Exiting.")
        sys.exit(1)
        
    np.random.shuffle(train_pickle_files) # Shuffle before splitting
    val_split_idx = int(len(train_pickle_files) * (1.0 - args_cli.val_split_percent))
    actual_train_files = train_pickle_files[:val_split_idx]
    val_files = train_pickle_files[val_split_idx:]

    if not actual_train_files:
        print("Error: No files left for training after validation split. Adjust val_split_percent or dataset size.")
        sys.exit(1)
    if not val_files and args_cli.val_split_percent > 0:
        print("Warning: No files for validation after split, though val_split_percent > 0.")


    dataset_train = EEGPickleDataset(
        filepaths=actual_train_files,
        expected_channels=args_cli.eeg_channels,
        expected_sfreq=args_cli.expected_sfreq,
        model_patch_length=args_cli.patch_length,
        map_filename_to_subject_id=not args_cli.no_subject_mapping, # store_true makes it True
        default_subject_id=0 # Can be an arg if needed
    )
    print(f"Training dataset size: {len(dataset_train)}")

    dataset_val = None
    if val_files:
        dataset_val = EEGPickleDataset(
            filepaths=val_files,
            expected_channels=args_cli.eeg_channels,
            expected_sfreq=args_cli.expected_sfreq,
            model_patch_length=args_cli.patch_length,
            map_filename_to_subject_id=not args_cli.no_subject_mapping,
            default_subject_id=0
        )
        print(f"Validation dataset size: {len(dataset_val)}")
    else:
        print("No validation dataset created.")

    train_sampler = None
    val_sampler = None
    if ddp:
        train_sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, drop_last=True
        )
        if dataset_val:
            val_sampler = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=False
            )
    
    # For IterableDataset, shuffle is part of the dataset's __iter__
    # For MapDataset (like EEGPickleDataset), DataLoader shuffle or sampler shuffle is needed.
    # If using DDP sampler, shuffle=False for DataLoader itself as sampler handles it.
    loader_shuffle_train = train_sampler is None 

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args_cli.batch_size,
        sampler=train_sampler,
        shuffle=loader_shuffle_train,
        num_workers=args_cli.num_workers,
        pin_memory=True,
        drop_last=True # Important for consistent batch sizes, esp. with DDP
    )
    data_loader_val = None
    if dataset_val:
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=args_cli.batch_size * 2, # Often larger for validation
            sampler=val_sampler,
            shuffle=False, # No shuffle for validation
            num_workers=args_cli.num_workers,
            pin_memory=True,
            drop_last=False # Can process all validation samples
        )
    print('Dataloaders prepared.')

    # --- Model Initialization ---
    iter_num = 0
    start_epoch = 0
    model_args_dict = get_model_args(args_cli)

    if args_cli.init_from == 'scratch':
        print("Initializing a new LBLM model from scratch\n\n\n")
        model = InnerSpeech_LBLM_MSTP(**model_args_dict)
    elif args_cli.init_from == 'resume':
        print("Resuming training from checkpoint\n\n\n")
        ckpt_path = args_cli.resume_ckpt_path
        if not ckpt_path:
            # Try to find the latest checkpoint in out_dir
            out_dir = args_cli.out_dir
            ckpt_files = []
            for fname in os.listdir(out_dir):
                if fname.startswith("ckpt_epoch") and fname.endswith(".pt"):
                    # Extract the epoch number
                    try:
                        num = int(fname[len("ckpt_epoch"):-3])
                        ckpt_files.append((num, fname))
                    except Exception:
                        continue
            if ckpt_files:
                # Pick the one with the largest epoch number
                ckpt_files.sort()
                latest_ckpt = ckpt_files[-1][1]
                ckpt_path = os.path.join(out_dir, latest_ckpt)
                print(f"Auto-selected latest checkpoint for resume: {ckpt_path}")
            else:
                # Try ckpt_final.pt as fallback
                final_ckpt = os.path.join(out_dir, "ckpt_final.pt")
                if os.path.exists(final_ckpt):
                    ckpt_path = final_ckpt
                    print(f"Auto-selected ckpt_final.pt for resume: {ckpt_path}")
                else:
                    print(f"No checkpoint found in {out_dir} to resume from.")
                    ckpt_path = ""
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"Warning: resume_ckpt_path '{ckpt_path}' not found. Initializing from scratch.")
            model = InnerSpeech_LBLM_MSTP(**model_args_dict)
        else:
            print(f"Resuming training from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Ensure checkpoint model_args are compatible or use them
            # For simplicity, we assume current CLI args define the model structure
            # and we are just loading weights.
            # If model_args were saved: model_args_dict_ckpt = checkpoint['model_args']
            # model = InnerSpeech_LBLM_MSTP(**model_args_dict_ckpt)
            model = InnerSpeech_LBLM_MSTP(**model_args_dict) # Using current args for structure
            
            # Fix for DDP-saved models: remove 'module.' prefix if it exists
            state_dict = checkpoint['model']
            if any(key.startswith('module.') for key in state_dict):
                print("Removing 'module.' prefix from checkpoint state_dict keys for non-DDP/single-GPU load.")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            try:
                model.load_state_dict(state_dict)
                iter_num = checkpoint.get('iter_num', 0)
                start_epoch = checkpoint.get('epoch', 0) + 1 # Start next epoch
                print(f"Resumed from iter_num: {iter_num}, start_epoch: {start_epoch}")
            except RuntimeError as e:
                print(f"Error loading state_dict: {e}. Model might be incompatible. Initializing from scratch.")
                model = InnerSpeech_LBLM_MSTP(**model_args_dict) # Fallback
                iter_num = 0
                start_epoch = 0
            del checkpoint # Free memory
    else:
        raise ValueError(f"Unknown init_from: {args_cli.init_from}")

    model.to(device)
    if ptdtype != torch.float32: # Ensure model params match ptdtype for AMP
        model = model.to(ptdtype)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


    optimizer = model.configure_optimizers(
        args_cli.weight_decay, 
        args_cli.learning_rate, 
        (args_cli.beta1, args_cli.beta2), 
        device_type,
        optimizer_type=args_cli.optimizer_type,
        eps=args_cli.lamb_eps # Pass eps here
    )
    if args_cli.init_from == 'resume' and ckpt_path and os.path.exists(ckpt_path):
        # Try to load optimizer state
        checkpoint_opt = torch.load(ckpt_path, map_location=device, weights_only=False) # Reload for optimizer
        if 'optimizer' in checkpoint_opt:
            try:
                optimizer.load_state_dict(checkpoint_opt['optimizer'])
                print("Optimizer state loaded from checkpoint.")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Initializing optimizer from scratch.")
        del checkpoint_opt

    scaler = torch.amp.GradScaler('cuda', enabled=(args_cli.dtype != 'float32'))

    # --- Compile and DDP Wrap ---
    if args_cli.compile and hasattr(torch, 'compile'):
        if master_process: print("Compiling the model... (takes a ~minute)")
        try:
            model = torch.compile(model)
            if master_process: print("Model compiled successfully.")
        except Exception as e:
            if master_process: print(f"Model compilation failed: {e}. Proceeding without compilation.")
    
    raw_model_for_saving = model # Before DDP wrap for saving state_dict easily
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model_for_saving = model.module # Get underlying model for saving

    # --- Learning Rate Scheduler ---
    num_training_steps_per_epoch = len(data_loader_train) # Batches per epoch
    lr_schedule_values = cosine_scheduler(
        args_cli.learning_rate, args_cli.min_lr, args_cli.epochs+start_epoch, num_training_steps_per_epoch,
        warmup_epochs=args_cli.warmup_epochs
    )

    # --- Training Loop ---
    if master_process: print("Starting training loop...")
    train_losses_log = {} # For accumulating losses over log_interval
    
    for epoch in range(start_epoch, args_cli.epochs):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch) # Needed for DDP sampler to shuffle correctly

        model.train()
        epoch_t0 = time.time()

        for step, batch in enumerate(data_loader_train):
            if not running: break # Check for interrupt signal

            # Determine and set the learning rate for this iteration
            current_iter_global = epoch * num_training_steps_per_epoch + step
            if args_cli.decay_lr and current_iter_global < len(lr_schedule_values):
                lr = lr_schedule_values[current_iter_global]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif not args_cli.decay_lr:
                lr = args_cli.learning_rate # Fixed LR
            else: # LR schedule ended or no decay
                lr = optimizer.param_groups[0]['lr'] # Keep last lr

            # Prepare batch
            x_eeg = batch['x_raw_eeg'].to(device, non_blocking=True)
            subject_ids = batch['subject_id'].to(device, non_blocking=True)
            if ptdtype != torch.float32:
                 x_eeg = x_eeg.to(ptdtype)
            # subject_ids are long, no need to cast to ptdtype

            # Forward backward update
            if ddp:
                model.require_backward_grad_sync = (step + 1) % args_cli.gradient_accumulation_steps == 0
            
            with ctx: # Autocast context
                loss, _, log = model(x_eeg, subject_ids=subject_ids) # Model returns total loss and log dict
                loss = loss / args_cli.gradient_accumulation_steps # Scale loss for accumulation
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss encountered at epoch {epoch}, step {step}. Skipping update.")
                # Potentially clear gradients and skip optimizer step
                optimizer.zero_grad(set_to_none=True)
                # Consider reducing LR or other recovery strategies if this happens often
                continue


            # Accumulate logs
            for k, v_tensor in log.items():
                v = v_tensor.item() # Get scalar value
                train_losses_log[k] = train_losses_log.get(k, []) + [v]

            scaler.scale(loss).backward()

            if (step + 1) % args_cli.gradient_accumulation_steps == 0:
                if args_cli.grad_clip > 0.0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args_cli.grad_clip)
                    

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Log LAMB stats (if using LAMB and wandb enabled)
                if args_cli.optimizer_type.lower() == 'lamb' and (step + 1) % args_cli.gradient_accumulation_steps == 0:
                    # Log per optimization step, not per micro-batch step
                    log_lamb_optimizer_stats_to_wandb(optimizer, global_step=iter_num) 

            iter_num +=1 # Global iteration counter

            # Logging 
            if (step + 1) % args_cli.log_interval == 0 and master_process:
                log_str = f"Epoch {epoch}/{args_cli.epochs} | Step {step+1}/{num_training_steps_per_epoch} | LR {lr:.2e} | "
                avg_losses_str = []
                for k, v_list in train_losses_log.items():
                    if v_list: # Check if list is not empty
                         avg_loss = sum(v_list) / len(v_list)
                         avg_losses_str.append(f"{k.replace('mstp/', '')} {avg_loss:.4f}")
                log_str += " | ".join(avg_losses_str)
                
                t1_step = time.time()
                batch_time = (t1_step - epoch_t0) / (step + 1) # Avg time per batch so far in epoch
                eta_seconds = batch_time * (num_training_steps_per_epoch - (step + 1))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                log_str += f" | BatchTime {batch_time*1000:.2f}ms | ETA {eta_str}"
                print(log_str)

                # Log scalars to W&B
                if args_cli.wandb_log:
                    wandb_log_data = {'learning_rate': lr, 'epoch': epoch, 'step_in_epoch': step + 1}
                    for k, v_list in train_losses_log.items():
                        if v_list:
                            wandb_log_data[k] = sum(v_list) / len(v_list) # Log averages
                    wandb.log(wandb_log_data, step=iter_num) # Use global iter_num as step
                train_losses_log = {} # Reset for next logging interval

            if not running: break # Inner loop break
        # End of epoch
        epoch_duration = time.time() - epoch_t0
        if master_process:
            print(f"Epoch {epoch} finished in {epoch_duration:.2f}s.")


        # Validation
        if data_loader_val and (epoch + 1) % args_cli.eval_interval == 0 and running:
            if master_process: print(f"Running validation for epoch {epoch}...")
            model.eval() # Set model to evaluation mode
            val_losses_accum = {}
            
            # Get one batch for plotting (if master_process)
            val_plot_batch_data = None
            if master_process and args_cli.wandb_log: # Only need data for plotting on master
                try:
                    val_plot_batch_data = next(iter(data_loader_val))
                except StopIteration:
                    print("Validation dataloader is empty, cannot get batch for plotting.")
                except Exception as e_plot_batch:
                    print(f"Error getting batch for plotting: {e_plot_batch}")

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(data_loader_val):
                    x_eeg_val = val_batch['x_raw_eeg'].to(device, non_blocking=True)
                    subject_ids_val = val_batch['subject_id'].to(device, non_blocking=True)
                    if ptdtype != torch.float32: # Use the global ptdtype for consistency
                        x_eeg_val = x_eeg_val.to(ptdtype)

                    with ctx: # Use the global ctx
                        val_loss, _, val_log = model(x_eeg_val, subject_ids=subject_ids_val)
                    
                    for k, v_tensor in val_log.items():
                        v = v_tensor.item()
                        val_losses_accum[k] = val_losses_accum.get(k, []) + [v]

            if master_process and val_losses_accum:
                val_log_str = f"Validation Epoch {epoch} | "
                avg_val_losses_str = []
                for k, v_list in val_losses_accum.items():
                     if v_list:
                        avg_loss = sum(v_list) / len(v_list)
                        avg_val_losses_str.append(f"{k.replace('mstp/', 'val_')} {avg_loss:.4f}")
                val_log_str += " | ".join(avg_val_losses_str)
                print(val_log_str)

                if args_cli.wandb_log:
                    wandb_val_log_data = {'val_epoch': epoch}
                    for k, v_list in val_losses_accum.items():
                        if v_list:
                            # Prefix with 'val/' for W&B to distinguish from train losses
                            wandb_k = f"val/{k.replace('mstp/', '')}" if 'mstp/' in k else f"val/{k}"
                            wandb_val_log_data[wandb_k] = sum(v_list) / len(v_list)
                    wandb.log(wandb_val_log_data, step=iter_num) # Log at the end of epoch / global step

                if val_plot_batch_data is not None and  (epoch + 1) % 7 == 0:
                    try:
                        # Plot with model's internal random masking for one channel
                        log_reconstruction_plot_to_wandb(
                            model, val_plot_batch_data, epoch, iter_num,
                            device, ptdtype, ctx, args_cli,
                            plot_masking_strategy='random' 
                        )
                        
                        # Plot with a harder block masking for one channel
                        log_reconstruction_plot_to_wandb(
                            model, val_plot_batch_data, epoch, iter_num,
                            device, ptdtype, ctx, args_cli,
                            plot_masking_strategy='block',
                            block_mask_size=5 # Or make this configurable via CLI
                        )
                        # print("Logged reconstruction plots to W&B.")
                    except Exception as e_plot:
                        print(f"Error during W&B plot logging: {e_plot}")
                        import traceback
                        traceback.print_exc()
            
            model.train() # Set back to train mode

        # Checkpoint saving
        if (epoch + 1) % args_cli.save_ckpt_freq == 0 and master_process and running:
            checkpoint_data = {
                'model': raw_model_for_saving.state_dict(), # Save unwrapped model
                'optimizer': optimizer.state_dict(),
                'model_args': model_args_dict, # Save model config
                'iter_num': iter_num,
                'epoch': epoch,
                'args_cli': args_cli # Save command line args
            }
            ckpt_file = os.path.join(args_cli.out_dir, f'ckpt_epoch{epoch}.pt')
            print(f"Saving checkpoint to {ckpt_file}")
            torch.save(checkpoint_data, ckpt_file)
            # Could also save a 'latest.pt'

        if not running: break # Outer loop break (after potential save)
    
    # End of training
    if master_process and running: # If training completed normally
        print("Training finished.")
        final_ckpt_file = os.path.join(args_cli.out_dir, 'ckpt_final.pt')
        print(f"Saving final checkpoint to {final_ckpt_file}")
        torch.save({
            'model': raw_model_for_saving.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args_dict,
            'iter_num': iter_num,
            'epoch': args_cli.epochs -1, # Last completed epoch
            'args_cli': args_cli
        }, final_ckpt_file)

    elif master_process and not running: # If interrupted
        print("Training interrupted. Final state might not be saved unless caught by signal handler save.")


    if ddp:
        destroy_process_group()

def get_cli_args():
    parser = argparse.ArgumentParser('LBLM MSTP Training Script')
    
    # Paths and Saving
    parser.add_argument('--out_dir', default='./out_lblm_mstp', type=str, help='Output directory for checkpoints and logs.')
    parser.add_argument('--dataset_dir', default='/workspace/tuh_full', type=str, help='Directory containing .pkl EEG segments.')
    parser.add_argument('--init_from', default='scratch', type=str, choices=['scratch', 'resume'], help="'scratch' or 'resume' from ckpt_path.")
    parser.add_argument('--resume_ckpt_path', default='', type=str, help='Path to checkpoint for resuming (if init_from=resume).')
    parser.add_argument('--log_interval', default=50, type=int, help="Log training stats every N steps.")
    parser.add_argument('--eval_interval', default=1, type=int, help="Evaluate on validation set every N epochs.")
    parser.add_argument('--save_ckpt_freq', default=1, type=int, help="Save checkpoint every N epochs.")

    # Data and Loader
    parser.add_argument('--val_split_percent', default=0.1, type=float, help="Percentage of data to use for validation (0.0 to 1.0).")
    parser.add_argument('--expected_sfreq', default=200, type=int, help="Expected sampling frequency of data in pickles (e.g., 200 for TUH script, 250 for LBLM paper).")
    parser.add_argument('--no_subject_mapping', action='store_true', help="Disable trying to map filenames to subject IDs in dataset.")
    parser.add_argument('--num_workers', default=4, type=int)

    # Training Hyperparameters
# Ensure these defaults match the LBLM paper for LAMB
    parser.add_argument('--optimizer_type', default='lamb', type=str.lower, choices=['lamb', 'adamw'], 
                        help="Optimizer type ('lamb' or 'adamw'). LBLM paper uses LAMB.")
    parser.add_argument('--learning_rate', default=1e-3, type=float, help="Initial learning rate (LBLM: 1e-3 for LAMB).")
    parser.add_argument('--weight_decay', default=0.01, type=float, help="Weight decay (LBLM: 0.01 for LAMB).")
    parser.add_argument('--beta1', default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument('--beta2', default=0.999, type=float, help="Optimizer beta2.")
    parser.add_argument('--lamb_eps', default=1e-6, type=float, help="Epsilon for LAMB optimizer (if different from AdamW).")
    parser.add_argument('--grad_clip', default=1.0, type=float, help="Gradient clipping value (LBLM: 1.0).")


    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=float, help="Minimum learning rate for cosine scheduler.")
    parser.add_argument('--decay_lr', default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to decay learning rate (cosine scheduler).")
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    
    # Model Architecture (LBLM specific - should match InnerSpeech_LBLM_MSTP defaults or be configurable)
    parser.add_argument('--patch_length', default=25, type=int)
    parser.add_argument('--patch_stride', default=6, type=int)
    parser.add_argument('--num_subjects', default=12, type=int, help="Max number of subjects for embedding (0 if no subject embedding). LBLM paper used 12.")
    parser.add_argument('--subject_embed_dim', default=1, type=int, help="Dimension for subject gain (1 for scalar).")
    parser.add_argument('--mask_ratio', default=0.15, type=float, help="Ratio of patches to mask for MSTP.")
    parser.add_argument('--no_rev_in', action='store_true', help="Disable Reversible Instance Normalization.")
    parser.add_argument('--eeg_channels', default=23, type=int, help="Number of EEG channels model expects (e.g., 60 after TUH processing, 122 for LBLM paper dataset).")
    
    parser.add_argument('--conformer_embed_dim', default=64, type=int, help="Embedding dim for patch embedder / input to conformer.")
    parser.add_argument('--conformer_layers', default=4, type=int)
    parser.add_argument('--conformer_d_model', default=64, type=int)
    parser.add_argument('--conformer_n_head', default=8, type=int)
    parser.add_argument('--conformer_d_ff', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout rate for conformer.")
    parser.add_argument('--conformer_conv_kernel', default=31, type=int)
    
    parser.add_argument('--huber_delta', default=1.0, type=float)
    parser.add_argument('--lambda_amplitude', default=0.1, type=float)
    parser.add_argument('--lambda_phase', default=0.1, type=float)

    # System and Performance
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--dtype', default='float32', type=str, choices=['float32', 'bfloat16', 'float16'], help="PyTorch dtype for training.")
    parser.add_argument('--compile', action='store_true', help="Enable torch.compile (PyTorch 2.0+).")
    parser.add_argument('--ddp_backend', default='nccl', type=str, help="DDP backend (e.g., nccl, gloo).")
    
    # Add wandb args if you plan to use it
    parser.add_argument('--wandb_log', action='store_true', help="Enable Weights & Biases logging.")
    parser.add_argument('--wandb_project', default='lblm_mstp_train', type=str, help="W&B project name.")
    parser.add_argument('--wandb_run_name', default='lblm_run_' + time.strftime("%Y%m%d_%H%M%S"), type=str, help="W&B run name.")



    return parser.parse_args()

if __name__ == '__main__':
    global args_cli
    args_cli = get_cli_args()
    signal.signal(signal.SIGINT, signal_handler_interrupt) # Register signal handler
    
    try:
        main_train_loop()
    except Exception as e:
        print(f"Unhandled exception in main_train_loop: {e}")
        import traceback
        traceback.print_exc()
        if ddp: # Ensure DDP cleanup if an error occurs after init
            if torch.distributed.is_initialized():
                destroy_process_group()
    finally:
        if args_cli.wandb_log and master_process:
            wandb.finish()
            print("Weights & Biases run finished.")
        if ddp:
            if torch.distributed.is_initialized():
                destroy_process_group()