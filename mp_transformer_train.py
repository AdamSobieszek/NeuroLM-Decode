# File: train_mp_tensorboard.py

import os
import time
import argparse
import signal
import sys
from contextlib import nullcontext
import glob
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io

try:
    from neural_mp.mp_transformer import MP_Model
    from neural_mp.synthetic_dataset import SyntheticMPDataset
    # from neural_mp.lblm_dataset import EEGPickleDataset
except ImportError as e:
    print(f"ImportError: {e}. Ensure model and dataset files are in the Python path.")
    sys.exit(1)

# --- Global DDP Variables ---
master_process = True
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
ptdtype = torch.float32
ctx = nullcontext()
ddp = False
running = True
writer = None

os.environ['TORCH_USE_CUDA_DSA'] = '1'

def signal_handler_interrupt(sig, frame):
    global running
    print('Keyboard interrupt received. Attempting to save and exit...')
    running = False

def fig_to_tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    from PIL import Image
    image = Image.open(buf)
    image_np = np.array(image.convert('RGB'))
    return torch.from_numpy(image_np).permute(2, 0, 1)

def log_mp_reconstruction_plot_to_tensorboard(model, val_batch, epoch, global_step):
    # This function remains unchanged, as it correctly logs the necessary plots.
    global args_cli, master_process, device, ptdtype, writer
    if not args_cli.tensorboard_log or not master_process or writer is None:
        return

    model.eval()
    x_eeg_val = val_batch['x_raw_eeg'].to(device, non_blocking=True).to(ptdtype)
    
    with torch.no_grad():
        # Make sure the model uses the full number of iterations for validation plots
        _model = model.module if ddp else model
        original_mp_iterations = _model.mp_iterations
        _model.mp_iterations = args_cli.mp_iterations
        
        _, final_residual, _ = model(x_eeg_val)
        
        # Restore original setting
        _model.mp_iterations = original_mp_iterations

    sample_idx = 0
    original_signal = x_eeg_val[sample_idx].cpu().float().numpy()
    final_residual = final_residual[sample_idx].cpu().float().numpy()
    reconstruction = original_signal - final_residual
    
    channels_to_plot = min(5, original_signal.shape[0])
    fig, axs = plt.subplots(channels_to_plot, 3, figsize=(18, 2.5 * channels_to_plot), sharex=True)
    fig.suptitle(f'Epoch {epoch} - MP Reconstruction & Residual', fontsize=16)
    for i in range(channels_to_plot):
        axs[i, 0].plot(original_signal[i], label=f'Channel {i}'); axs[i, 0].set_title('Original Signal'); axs[i, 0].legend(loc='upper right'); axs[i, 0].grid(True, linestyle='--', alpha=0.6)
        axs[i, 1].plot(reconstruction[i]); axs[i, 1].set_title('Reconstruction')
        axs[i, 2].plot(final_residual[i]); axs[i, 2].set_title('Final Residual')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    writer.add_image("Val/Reconstruction", fig_to_tensor(fig), global_step)
    plt.close(fig)

    _model = model.module if ddp else model
    num_atoms_to_plot = min(16, _model.mp_layer.num_atoms)
    atoms = _model.mp_layer.dictionary.detach().cpu().float()
    atoms = atoms.view(_model.mp_layer.num_atoms, _model.num_channels, _model.signal_length)
    fig_atoms, axs_atoms = plt.subplots(4, 4, figsize=(12, 8))
    fig_atoms.suptitle(f'Learned Atoms (Epoch {epoch})', fontsize=16)
    axs_atoms = axs_atoms.flatten()
    for i in range(num_atoms_to_plot):
        axs_atoms[i].plot(atoms[i, 0, :].numpy()); axs_atoms[i].set_title(f'Atom {i}', fontsize=8); axs_atoms[i].set_xticks([]); axs_atoms[i].set_yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    writer.add_image("Val/Learned_Atoms", fig_to_tensor(fig_atoms), global_step)
    plt.close(fig_atoms)
    model.train()

def init_ddp(args_ddp_backend):
    # This function remains unchanged.
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, ptdtype, ddp_local_rank, args_cli
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=args_ddp_backend)
        ddp_rank = int(os.environ['RANK']); ddp_local_rank = int(os.environ['LOCAL_RANK']); ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'; torch.cuda.set_device(device); master_process = ddp_rank == 0
    else:
        ddp_world_size, ddp_rank, ddp_local_rank = 1, 0, 0
    torch.manual_seed(args_cli.seed + ddp_rank)
    if device == 'cuda': torch.cuda.manual_seed(args_cli.seed + ddp_rank)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args_cli.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if 'cuda' in device else nullcontext()
    if master_process: print(f"Running on {device} with dtype {args_cli.dtype}. DDP: {ddp}")

# --- NEW: Learning Rate Scheduler Function ---
def get_lr_schedule(warmup_epochs, epochs, niter_per_ep, base_lr, min_lr):
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep
    
    warmup_schedule = np.linspace(min_lr, base_lr, warmup_iters) if warmup_iters > 0 else np.array([])
    
    decay_iters = np.arange(total_iters - warmup_iters)
    cosine_schedule = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(math.pi * decay_iters / (total_iters - warmup_iters)))
    
    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    return schedule

def main_train_loop():
    global running, master_process, device, ptdtype, ctx, ddp, args_cli, writer

    init_ddp(args_cli.ddp_backend)

    if master_process and args_cli.tensorboard_log:
        log_dir = os.path.join(args_cli.out_dir, 'logs', args_cli.run_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging enabled. Logs will be saved to: {log_dir}")

    # --- Dataloader Setup (Preserving your logic) ---
    signal_length = args_cli.signal_duration * args_cli.expected_sfreq
    if args_cli.use_synthetic_data:
        if master_process: print("Using Synthetic Dataset for training and validation.")
        dataset_train = SyntheticMPDataset(num_samples=args_cli.batch_size*300, num_channels=args_cli.eeg_channels, signal_length=signal_length, num_true_atoms=args_cli.num_atoms//2, sparsity=4, noise_level=0.0)
        dataset_val = SyntheticMPDataset(num_samples=args_cli.batch_size*3, num_channels=args_cli.eeg_channels, signal_length=signal_length, num_true_atoms=args_cli.num_atoms//2, sparsity=4, noise_level=0.0)
    else:
        # ... your logic for loading real data would go here ...
        pass

    if master_process:
        print(f"Train dataset size: {len(dataset_train)}, Val dataset size: {len(dataset_val)}")
        print(f"Model will expect signals of shape: [{args_cli.eeg_channels}, {signal_length}]")

    train_sampler = DDP(dataset_train) if ddp else None
    data_loader_train = DataLoader(dataset_train, batch_size=args_cli.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args_cli.num_workers, pin_memory=True, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args_cli.batch_size * 2, shuffle=False, num_workers=args_cli.num_workers, pin_memory=True)

    iter_num, start_epoch = 0, 0
    model_args = {'num_channels': args_cli.eeg_channels, 'signal_length': signal_length, 'num_atoms': args_cli.num_atoms, 'mp_iterations': args_cli.mp_iterations, 'attention_temp': args_cli.attention_temp}
    model = MP_Model(**model_args)
    model.to(device)
    if master_process: print(f"MP_Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    optimizer = model.configure_optimizers(args_cli.weight_decay, args_cli.learning_rate, (args_cli.beta1, args_cli.beta2), device)
    scaler = torch.amp.GradScaler(enabled=(args_cli.dtype == 'float16'))
    
    if args_cli.compile:
        if master_process: print("Compiling the model...")
        model = torch.compile(model)
    if ddp: model = DDP(model, device_ids=[ddp_local_rank])

    # --- NEW: Initialize LR Scheduler ---
    lr_schedule = None
    if args_cli.decay_lr:
        lr_schedule = get_lr_schedule(args_cli.warmup_epochs, args_cli.epochs, len(data_loader_train), args_cli.learning_rate, args_cli.min_lr)
        if master_process: print("Using Cosine Decay LR scheduler with warmup.")

    if master_process: print("Starting MP training loop...")
    for epoch in range(start_epoch, args_cli.epochs):
        if ddp and train_sampler is not None: train_sampler.set_epoch(epoch)
        model.train()
        # model.mp_layer.temperature.data = torch.tensor(min(0.0001, model.mp_layer.temperature.data*0.9))
        for step, batch in enumerate(data_loader_train):
            if not running: break
            
            # --- NEW: Set LR based on schedule ---
            if lr_schedule is not None:
                lr = lr_schedule[iter_num]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = args_cli.learning_rate
            
            x_eeg = batch['x_raw_eeg'].to(device, non_blocking=True).to(ptdtype)
            
            # --- Preserving your training step logic ---
            # This loop performs one optimization step per iteration.
            with ctx:
                x_eeg = x_eeg.detach()
                reconstruction_loss, x_eeg, log = model(x_eeg)
            
            # --- NEW: Calculate and add diversity penalty ---
            diversity_loss = torch.tensor(0.0, device=device)
            if args_cli.diversity_penalty_weight > 0:
                _model = model.module if ddp else model
                atoms = _model.mp_layer.dictionary
                # Normalize atoms to unit vectors for cosine similarity
                atoms_norm = F.normalize(atoms, p=2, dim=1)
                # Compute cosine similarity matrix: (N_atoms, D) @ (D, N_atoms) -> (N_atoms, N_atoms)
                similarity_matrix = torch.matmul(atoms_norm, atoms_norm.t())
                # Penalize off-diagonal elements. We want them to be close to 0.
                # Taking the mean of the absolute values of the off-diagonal elements.
                num_atoms = atoms.shape[0]
                # Subtract identity matrix to zero out the diagonal
                off_diagonal_sim = similarity_matrix - torch.eye(num_atoms, device=device)
                diversity_loss = torch.mean(torch.abs(off_diagonal_sim))
            
            total_loss = reconstruction_loss + args_cli.diversity_penalty_weight * diversity_loss
                
            scaler.scale(total_loss).backward()
            if args_cli.grad_clip > 0.0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), args_cli.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            iter_num += 1

            if iter_num % args_cli.log_interval == 0 and master_process:
                print(f"Epoch {epoch} | Step {step} | Total Loss: {total_loss.item():.6f} (Recon: {reconstruction_loss.item():.6f}, Diversity: {diversity_loss.item():.4f}) | LR: {lr:.2e}")
                if writer:
                    writer.add_scalar('train/total_loss', total_loss.item(), iter_num)
                    writer.add_scalar('train/reconstruction_loss', reconstruction_loss.item(), iter_num)
                    writer.add_scalar('train/diversity_loss', diversity_loss.item(), iter_num)
                    writer.add_scalar('train/learning_rate', lr, iter_num)
        
        if not running: break

        # Validation
        if (epoch + 1) % args_cli.eval_interval == 0:
            model.eval()
            val_loss_total = 0.0
            val_plot_batch = None
            with torch.no_grad():
                for val_batch in data_loader_val:
                    if val_plot_batch is None and master_process: val_plot_batch = val_batch
                    x_eeg_val = val_batch['x_raw_eeg'].to(device, non_blocking=True).to(ptdtype)
                    loss_val, _, _ = model(x_eeg_val)
                    val_loss_total += loss_val.item()
            
            avg_val_loss = val_loss_total / len(data_loader_val)
            if master_process:
                print(f"--- Validation Epoch {epoch} | Avg Loss: {avg_val_loss:.6f} ---")
                if writer:
                    writer.add_scalar('val/loss', avg_val_loss, iter_num)
                    if val_plot_batch:
                        log_mp_reconstruction_plot_to_tensorboard(model, val_plot_batch, epoch, iter_num)
            model.train()

        if not running: break
    
    if ddp: destroy_process_group()

def get_cli_args():
    parser = argparse.ArgumentParser('Matching Pursuit Training Script with TensorBoard')
    
    # Paths and Saving (Preserving your defaults)
    parser.add_argument('--out_dir', default='./out_mp', type=str)
    parser.add_argument('--dataset_dir', default='', type=str, help="Required if not using synthetic data")
    parser.add_argument('--run_name', default='mp_run_' + time.strftime("%Y%m%d_%H%M%S"), type=str, help="Name for the run, used in log directory.")
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)

    # Data (Preserving your defaults)
    parser.add_argument('--val_split_percent', default=0.1, type=float)
    parser.add_argument('--eeg_channels', default=4, type=int)
    parser.add_argument('--expected_sfreq', default=100, type=int)
    parser.add_argument('--signal_duration', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--use_synthetic_data', action='store_true')

    # MP Model Hyperparameters (Preserving your defaults)
    parser.add_argument('--num_atoms', default=128, type=int)
    parser.add_argument('--mp_iterations', default=5, type=int)
    parser.add_argument('--attention_temp', default=.5, type=float)

    # Training Hyperparameters (Preserving your defaults and adding new ones)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    
    # --- NEW: LR Scheduler and Diversity Penalty Arguments ---
    parser.add_argument('--decay_lr', action='store_true', help="Enable cosine decay learning rate scheduler.")
    parser.add_argument('--warmup_epochs', default=1, type=int, help="Number of warmup epochs for LR scheduler.")
    parser.add_argument('--min_lr', default=1e-5, type=float, help="Minimum learning rate for cosine decay.")
    parser.add_argument('--diversity_penalty_weight', default=0.2, type=float, help="Weight for the atom diversity penalty. 0 to disable.")

    # System (Preserving your defaults)
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--dtype', default='float32', type=str, choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--ddp_backend', default='nccl', type=str)
    parser.add_argument('--tensorboard_log', action='store_true', help="Enable TensorBoard logging.")
        
    return parser.parse_args()

if __name__ == '__main__':
    args_cli = get_cli_args()
    signal.signal(signal.SIGINT, signal_handler_interrupt)
    
    try:
        main_train_loop()
    except Exception as e:
        print(f"Unhandled exception in main_train_loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ddp and torch.distributed.is_initialized():
            destroy_process_group()
        if writer:
            writer.close()
            if master_process:
                print("TensorBoard writer closed.")