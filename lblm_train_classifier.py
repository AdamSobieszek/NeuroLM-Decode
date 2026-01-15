# File: lblm_train_classifier.py

import os
import time
import argparse
import signal
import sys
from contextlib import nullcontext
import glob

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import collections
import copy ### NEW

# --- New imports for classification logging ---
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Project-specific imports ---
try:
    # We need the model, dataset, and the new classifier
    from model2 import InnerSpeech_LBLM_MSTP
    from model2.st_classifier_v2 import SpatioTemporalClassifier
    from lblm_dataset import EEGPickleDataset 
except ImportError as e:
    print(f"ImportError: {e}. Ensure model, dataset, and classifier files are correctly set up.")
    sys.exit(1)

# --- Global DDP Variables ---
master_process = True
torch.set_float32_matmul_precision("high") 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = torch.bfloat16
ctx = nullcontext()
ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
device_type = 'cuda' if 'cuda' in device else 'cpu'
ddp = False
running = True

# -----------------------------------------------------------
# 2025 RECIPE CONSTANTS (feel free to tweak in CLI)
# -----------------------------------------------------------
HEAD_LR                = 3e-4
BACKBONE_LR            = 3e-5            # 0.1 × head LR
WEIGHT_DECAY_HEAD      = 0.01
WEIGHT_DECAY_BACKBONE  = 0.00
BETAS                  = (0.9, 0.95)
FREEZE_EPOCHS          = 2              # keep backbone frozen for n epochs
EMA_WEIGHT_DECAY       = 0.05           # EMA of weights

# --- NEW: Classification Logging Function ---
def log_classification_artifacts_to_wandb(model, val_loader, epoch, global_step, device, ctx, args_cli_val):
    """
    Calculates validation accuracy, generates a confusion matrix, and logs them to W&B.
    """
    if not args_cli_val.wandb_log or not master_process:
        return

    was_training = model.training
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x_eeg = batch['x_raw_eeg'].to(device, non_blocking=True).to(ptdtype)
            subject_ids = batch['subject_id'].to(device, non_blocking=True)
            labels = batch['target'].to(device, non_blocking=True)

            with ctx:
                # The model's forward pass now returns classification loss and logs
                # We need the model's raw output (logits) to get predictions
                _model_ref = model.module if ddp else model
                loss, logits, log = _model_ref(x_eeg, subject_ids, labels=labels)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # --- Create and Log Confusion Matrix ---
    if not all_labels or not all_preds:
        print("Not enough data to generate confusion matrix.")
        if was_training: model.train()
        return

    # Remove ignored labels (-1) if any
    valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
    all_labels = [all_labels[i] for i in valid_indices]
    all_preds = [all_preds[i] for i in valid_indices]

    if not all_labels:
        print("No valid labels found for confusion matrix.")
        if was_training: model.train()
        return

    # Calculate overall accuracy
    val_accuracy = accuracy_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(args_cli_val.num_classes)))
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'Epoch {epoch} | Validation Confusion Matrix | Acc: {val_accuracy:.4f}')

    try:
        wandb.log({
            "Val Viz/Confusion Matrix": wandb.Image(plt),
            "val/accuracy_from_cm": val_accuracy
        }, step=global_step)
    except Exception as e:
        print(f"Error logging confusion matrix to W&B: {e}")
    finally:
        plt.close(fig)

    if was_training: model.train()


# --- Signal Handler ---
def signal_handler_interrupt(sig, frame):
    global running
    print('Keyboard interrupt received. Attempting to save checkpoint and exit...')
    running = False

def init_ddp(args_ddp_backend):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, ptdtype, device_type, ddp_local_rank, args_cli
    ddp_flag = int(os.environ.get('RANK', -1)) != -1
    if ddp_flag:
        init_process_group(backend=args_ddp_backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        ddp = True
        print(f"DDP initialized: Rank {ddp_rank}/{ddp_world_size}")
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp = False
        print(f"Running in non-DDP mode on device: {device}")
    torch.manual_seed(args_cli.seed + seed_offset)
    if device == 'cuda':
        torch.cuda.manual_seed(args_cli.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args_cli.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    print(f"Using dtype: {args_cli.dtype}, Context: {type(ctx)}")

# --- Helper to reconstruct model args ---
def get_model_args(args_cli):
    """Creates a dictionary of model parameters from CLI args."""
    # This function is now crucial for re-instantiating the backbone correctly
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
        # "huber_delta": args_cli.huber_delta,
        # "lambda_amplitude": args_cli.lambda_amplitude,
        # "lambda_phase": args_cli.lambda_phase,
        # "lambda_word": args_cli.lambda_word
    }

def main_train_loop():
    global running, master_process, device, ptdtype, ctx, ddp, ddp_world_size, ddp_rank, args_cli

    init_ddp(args_cli.ddp_backend)

    if master_process:
        os.makedirs(args_cli.out_dir, exist_ok=True)
        if args_cli.wandb_log:
            try:
                wandb_run = wandb.init(project=args_cli.wandb_project, name=args_cli.wandb_run_name, config=args_cli, resume="allow")
                print(f"Weights & Biases initialized. Run name: {wandb_run.name}")
            except Exception as e:
                print(f"Error initializing W&B: {e}. Disabling logging.")
                args_cli.wandb_log = False

    # --- Dataloader Setup ---
    print('Preparing dataloaders for classification...')
    if args_cli.dataset_dir.startswith('[') and args_cli.dataset_dir.endswith(']'):
        dataset_dirs = eval(args_cli.dataset_dir)
        train_pickle_files = []
        for d in dataset_dirs:
            train_pickle_files.extend(glob.glob(os.path.join(d, "**", "*.pkl"), recursive=True))
    else:
        train_pickle_files = glob.glob(os.path.join(args_cli.dataset_dir, "**", "*.pkl"), recursive=True)


    if not train_pickle_files:
        print(f"Error: No pickle files found in {args_cli.dataset_dir}. Exiting.")
        sys.exit(1)
        
    rng = np.random.default_rng(args_cli.seed)
    rng.shuffle(train_pickle_files)
    val_split_idx = int(len(train_pickle_files) * (1.0 - args_cli.val_split_percent))
    actual_train_files = train_pickle_files[:val_split_idx]
    val_files = train_pickle_files[val_split_idx:]

    dataset_train = EEGPickleDataset(
        filepaths=actual_train_files, expected_channels=args_cli.eeg_channels, desired_processing_duration_sec=2.8,
        expected_sfreq=args_cli.expected_sfreq, map_filename_to_subject_id=True,
        apply_multi_band_mixing=True, iterate_all_bands_per_file=True, load_classification_label=True
    )
    dataset_val = EEGPickleDataset(
        filepaths=val_files*2, expected_channels=args_cli.eeg_channels, desired_processing_duration_sec=2.8,
        expected_sfreq=args_cli.expected_sfreq, map_filename_to_subject_id=True,
        apply_multi_band_mixing=True, iterate_all_bands_per_file=True, load_classification_label=True
    ) if val_files else None

    train_sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True) if ddp else None
    val_sampler = torch.utils.data.DistributedSampler(dataset_val, shuffle=False) if ddp and dataset_val else None

    data_loader_train = DataLoader(dataset_train, batch_size=args_cli.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args_cli.num_workers, pin_memory=True, persistent_workers=args_cli.num_workers > 0, prefetch_factor=2, drop_last=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args_cli.batch_size, sampler=val_sampler, shuffle=False, num_workers=args_cli.num_workers, pin_memory=True, persistent_workers=args_cli.num_workers > 0, prefetch_factor=2, drop_last=False) if dataset_val else None
    print('Dataloaders prepared.')

    # --- Model Initialization ---
    iter_num = 0
    start_epoch = 0
    
    if args_cli.init_from != 'resume':
        raise ValueError("Classifier training must start from a pre-trained model. Use --init_from 'resume'.")

    print(f"Loading pre-trained backbone from: {args_cli.resume_ckpt_path}")
    checkpoint = torch.load(args_cli.resume_ckpt_path, weights_only=False, map_location=device)
    
    # Re-create model args from CLI to ensure correct structure
    model_args_dict = get_model_args(args_cli)

    classifier = SpatioTemporalClassifier(
        num_classes=args_cli.num_classes,
        d_model_from_backbone=model_args_dict["conformer_d_model"],
        num_eeg_channels=model_args_dict["eeg_channels"],
    )
    model = InnerSpeech_LBLM_MSTP(**model_args_dict, classifier_head=classifier)
    
    state_dict = checkpoint['model']
    if any(key.startswith('module.') for key in state_dict) or True:
        # print("Removing 'module.' prefix from checkpoint state_dict keys for non-DDP/single-GPU load.")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        old_state_dict = model.state_dict()
        for k, v in state_dict.items():
            k = k[7:] if k.startswith('module.') else k
            new_state_dict[k] = v
            if k in old_state_dict:
                if v.shape!=old_state_dict[k].shape:
                    print(f"Warning: Removed {k} due to shape mismatch: {v.shape} != {old_state_dict[k].shape}")
                    del new_state_dict[k]
                    load_optimizer_state = False
            else:
                print(f"Warning: Removed {k} due to not in old_state_dict")
                del new_state_dict[k]
                load_optimizer_state = False

        state_dict = new_state_dict
            
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # state_dict = {k: v for k, v in state_dict.items() if "classifier_head" not in k}
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pre-trained backbone. Missing keys (expected classifier): {missing_keys}")
    
    if args_cli.reset_iter_num:
        iter_num = 0
        start_epoch = 0
    else:
        iter_num = checkpoint.get('iter_num', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1

    model.to(device)
    if ptdtype != torch.float32: model = model.to(ptdtype)
    
    model.freeze_backbone(freeze=args_cli.freeze_backbone and FREEZE_EPOCHS > start_epoch)
    
    print(f"Model ready. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    ### NEW: two-group AdamW (head/backbone) ------------------------------
    def build_optimizer(model):
        head_params, backbone_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:                 # frozen?
                continue
            (head_params if "classifier_head" in n else backbone_params).append(p)

        return torch.optim.AdamW(
            [
                {"params": head_params,
                 "lr": HEAD_LR,
                 "weight_decay": WEIGHT_DECAY_HEAD},
                {"params": backbone_params,
                 "lr": BACKBONE_LR,
                 "weight_decay": WEIGHT_DECAY_BACKBONE},
            ],
            betas=BETAS)

    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', enabled=(args_cli.dtype == 'float16'))

    ### NEW: warm-up + cosine scheduler ------------------------------
    steps_per_epoch = math.ceil(len(data_loader_train) / args_cli.gradient_accumulation_steps)
    total_steps     = steps_per_epoch * args_cli.epochs
    warmup_steps    = int(0.1 * total_steps)               # 5 %

    def lr_lambda(step):
        step = step
        if step < warmup_steps:
            return step / warmup_steps                       # linear warm-up
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * prog))        # cosine 1→0
        return 0.01 + 0.99 * cosine                          # floor at 1 %

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if args_cli.compile and hasattr(torch, 'compile'):
        print("Compiling the model..."); model = torch.compile(model)
    
    raw_model_for_saving = model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=args_cli.freeze_backbone)
        raw_model_for_saving = model.module

    ### NEW: keep an EMA copy for evaluation / checkpoint ------------
    if not args_cli.no_ema_eval:    
        ema_model = copy.deepcopy(raw_model_for_saving)
        for p in ema_model.parameters():
            p.requires_grad = False

    # --- Training Loop ---
    print("Starting classifier training loop...")
    for epoch in range(start_epoch, args_cli.epochs):
        if ddp and train_sampler is not None: train_sampler.set_epoch(epoch)
        
        ### NEW: un-freeze backbone at epoch = FREEZE_EPOCHS ---------
        if epoch >= FREEZE_EPOCHS and args_cli.freeze_backbone and model.backbone_frozen:
            if master_process:
                print(f"\n--- Epoch {epoch}: Unfreezing backbone ---\n")
            raw_model_for_saving.freeze_backbone(freeze=False)
            optimizer = build_optimizer(raw_model_for_saving)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            if ddp:
                model.find_unused_parameters = False

        model.train()
        train_log_accum = collections.defaultdict(list)
        
        for step, batch in enumerate(data_loader_train):
            if not running: break
            lr = optimizer.param_groups[0]['lr']
            x_eeg = batch['x_raw_eeg'].to(device, non_blocking=True).to(ptdtype)
            subject_ids = batch['subject_id'].to(device, non_blocking=True)
            labels = batch['target'].to(device, non_blocking=True)
            bands_applied = batch['band_applied']

            with ctx:
                loss, logits, log = model(x_eeg, subject_ids=subject_ids, labels=labels, input_kwargs={'bands_applied': bands_applied})
                loss = loss / args_cli.gradient_accumulation_steps
            
            for k, v in log.items(): train_log_accum[k].append(v.item())
            scaler.scale(loss).backward()

            if (step + 1) % args_cli.gradient_accumulation_steps == 0:
                if args_cli.grad_clip > 0.0:
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), args_cli.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                # if not args_cli.no_ema_eval:
                #     ### NEW: EMA update ---------------------------------
                with torch.no_grad():
                    for p, ema_p in zip(raw_model_for_saving.named_parameters(), ema_model.named_parameters()):
                        if 'classifier_head' not in p[0]:
                            p[1].lerp_(ema_p[1], EMA_WEIGHT_DECAY)
                scheduler.step()
            iter_num += 1

            if (step + 1) % args_cli.log_interval == 0 and master_process:
                avg_loss = np.mean(train_log_accum['classifier/loss'])
                avg_acc = np.mean(train_log_accum['classifier/accuracy'])
                
                # Correctly calculate weighted per-band accuracy for training
                band_accs = {}
                for k in sorted([*train_log_accum.keys()]):
                    if '_accuracy' in k and 'classifier/' in k:
                        n_samples_key = k.replace('_accuracy', '_n_samples')
                        if n_samples_key in train_log_accum:
                            total_correct = sum(acc * n_samp for acc, n_samp in zip(train_log_accum[k], train_log_accum[n_samples_key]))
                            total_samples = sum(train_log_accum[n_samples_key])
                            weighted_avg_acc = total_correct / total_samples if total_samples > 0 else np.nan
                            log_key = k.replace('classifier/', 'train/').replace('accuracy', 'acc')
                            band_accs[log_key] = weighted_avg_acc
                
                print(f"Epoch {epoch} | Step {step+1}/{len(data_loader_train)} | LR {lr:.2e} | Train Loss {avg_loss:.4f} | Train Acc {avg_acc:.4f}\n" + (f"" + " ".join([f"{k.replace('train/', '')}: {v:.3f}" for k, v in band_accs.items()]) if band_accs else ""))
                if args_cli.wandb_log: wandb.log({'train/loss': avg_loss, 'train/accuracy': avg_acc, 'learning_rate': lr, 'epoch': epoch, **band_accs}, step=iter_num)
                train_log_accum.clear()
            if not running: break
        
        if data_loader_val and (epoch + 1) % args_cli.eval_interval == 0 and running:
            ### NEW: swap in EMA weights for validation -------------------
            swap_model = raw_model_for_saving# ema_model if not args_cli.no_ema_eval else raw_model_for_saving
            swap_model.eval()
            val_log_accum = collections.defaultdict(list)
            with torch.no_grad():
                for val_batch in data_loader_val:
                    x_eeg_val, s_ids_val, labels_val, bands_applied = val_batch['x_raw_eeg'].to(device, non_blocking=True).to(ptdtype), val_batch['subject_id'].to(device, non_blocking=True), val_batch['target'].to(device, non_blocking=True), val_batch['band_applied']
                    
                    with ctx: _, _, val_log = swap_model(x_eeg_val, subject_ids=s_ids_val, labels=labels_val, input_kwargs={'bands_applied': bands_applied})
                    for k, v in val_log.items(): val_log_accum[k].append(v.item())
            
            if master_process and val_log_accum:
                avg_val_loss, avg_val_acc = np.mean(val_log_accum['classifier/loss']), np.mean(val_log_accum['classifier/accuracy'])

                # Calculate weighted per-band accuracy for validation
                val_band_accs = {}
                for k in sorted([*val_log_accum.keys()]):
                    if '_accuracy' in k and 'classifier/' in k:
                        n_samples_key = k.replace('_accuracy', '_n_samples')
                        if n_samples_key in val_log_accum:
                            total_correct = sum(acc * n_samp for acc, n_samp in zip(val_log_accum[k], val_log_accum[n_samples_key]))
                            total_samples = sum(val_log_accum[n_samples_key])
                            weighted_avg_acc = total_correct / total_samples if total_samples > 0 else np.nan
                            log_key = k.replace('classifier/', 'val/').replace('accuracy', 'acc')
                            val_band_accs[log_key] = weighted_avg_acc
                print(f"Validation Epoch {epoch} | Val Loss {avg_val_loss:.4f} | Val Acc {avg_val_acc:.4f}\n" + (f"" + " ".join([f"{k.replace('val/', '')}: {v:.3f}" for k, v in val_band_accs.items()]) if val_band_accs else ""))
                if args_cli.wandb_log:
                    wandb.log({'val/loss': avg_val_loss, 'val/accuracy': avg_val_acc, **val_band_accs}, step=iter_num)
                    log_classification_artifacts_to_wandb(swap_model, data_loader_val, epoch, iter_num, device, ctx, args_cli)
        
        if master_process and (epoch) % args_cli.save_ckpt_freq == args_cli.save_ckpt_freq-1 and running:
            ckpt_file = os.path.join(args_cli.out_dir, f'ckpt_classifier_epoch{epoch}.pt')
            print(f"Saving checkpoint to {ckpt_file}")
            # CHANGED: Save EMA model if use_ema_eval is true for consistency
            model_to_save = raw_model_for_saving# ema_model if not args_cli.no_ema_eval else raw_model_for_saving
            torch.save({'model': model_to_save.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args_cli': args_cli}, ckpt_file)
            
        if not running: break

    if master_process and running:
        final_ckpt_file = os.path.join(args_cli.out_dir, 'ckpt_classifier_final.pt')
        print(f"Saving final checkpoint to {final_ckpt_file}"); 
        # CHANGED: Save EMA model as final checkpoint if use_ema_eval is true
        model_to_save = raw_model_for_saving#ema_model if not args_cli.no_ema_eval else raw_model_for_saving
        torch.save({'model': model_to_save.state_dict()}, final_ckpt_file)
    if ddp: destroy_process_group()

def get_cli_args():
    parser = argparse.ArgumentParser('LBLM Classifier Training Script')
    
    # Paths and Saving
    parser.add_argument('--out_dir', default='./out_lblm_classifier', type=str)
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--init_from', default='resume', type=str, choices=['resume'])
    parser.add_argument('--resume_ckpt_path', required=True, type=str, help='Path to PRE-TRAINED backbone checkpoint.')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--eval_interval', default=3, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int, help="Save checkpoint every N epochs.")

    # Data and Loader
    parser.add_argument('--val_split_percent', default=0.1, type=float)
    parser.add_argument('--eeg_channels', default=62, type=int)
    parser.add_argument('--expected_sfreq', default=200, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # Classification Task
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--freeze_backbone', action=argparse.BooleanOptionalAction, default=False)
    # parser.add_argument('--unfreeze_epoch', default=5, type=int)

    # Training Hyperparameters
    parser.add_argument('--learning_rate', default=HEAD_LR, type=float)           # kept for backward-compat ### CHANGED
    parser.add_argument('--weight_decay', default=WEIGHT_DECAY_HEAD, type=float)  # idem ### CHANGED
    parser.add_argument('--beta1', default=BETAS[0], type=float) ### CHANGED
    parser.add_argument('--beta2', default=BETAS[1], type=float) ### CHANGED
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--optimizer_type', default='lamb', type=str.lower, choices=['lamb', 'adamw']) # REMOVED: can be removed, now hard-wired to AdamW
    parser.add_argument('--warmup_epochs', default=0, type=int) # REMOVED: can be removed, now hard-wired to 0
    parser.add_argument('--no_ema_eval', action=argparse.BooleanOptionalAction, default=False) ### NEW

    # System and Performance
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--dtype', default='bfloat16', type=str, choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--ddp_backend', default='nccl', type=str)
    parser.add_argument('--reset_iter_num', action='store_true')
    
    # W&B Logging
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--wandb_project', default='lblm_classifier_train', type=str)
    parser.add_argument('--wandb_run_name', default='clf_run_' + time.strftime("%Y%m%d_%H%M%S"), type=str)
    
    # Backbone Architecture (for loading pre-trained model)
    parser.add_argument('--patch_length', default=20, type=int)
    parser.add_argument('--patch_stride', default=15, type=int)
    parser.add_argument('--num_subjects', default=128, type=int)
    parser.add_argument('--subject_embed_dim', default=64, type=int)
    parser.add_argument('--mask_ratio', default=0.0, type=float)
    parser.add_argument('--no_rev_in', action='store_true')
    parser.add_argument('--conformer_embed_dim', default=64, type=int)
    parser.add_argument('--conformer_layers', default=4, type=int)
    parser.add_argument('--conformer_d_model', default=512, type=int)
    parser.add_argument('--conformer_n_head', default=8, type=int)
    parser.add_argument('--conformer_d_ff', default=512, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--conformer_conv_kernel', default=31, type=int)
    parser.add_argument('--huber_delta', default=1.0, type=float)
    parser.add_argument('--lambda_amplitude', default=0.1, type=float)
    parser.add_argument('--lambda_phase', default=0.1, type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    global args_cli
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
        if 'args_cli' in globals() and args_cli.wandb_log and master_process:
            wandb.finish()