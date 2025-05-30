
"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import os
import time
import argparse
import signal
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.model_inner_speech_transformer import VQ_Align
from model.model_neural_transformer import NTConfig
# from InnerSpeechMVPv1.src.data_utils.neurolm_dataloaders import ThinkingOutLoudLoader as PickleLoader
from dataset import PickleLoader
from pathlib import Path
from utils import cosine_scheduler
import math


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None
running = True


def signal_handler(sig, frame):
    global running
    print('Keyboard interrupt received. Saving checkpoint and exiting...')
    running = False

def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank,dtype
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
     #'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.xenviron['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    global ptdtype
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def save_checkpoint(raw_model, optimizer, encoder_args, decoder_args, iter_num, epoch, checkpoint_out_dir):
    if master_process:
        # First detach model from GPU temporarily to free memory
        model_state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
        optimizer_state = optimizer.state_dict()  # This should be small
        
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer_state,
            'encoder_args': encoder_args,
            'decoder_args': decoder_args,
            'iter_num': iter_num,
            'epoch': epoch
        }
        print(f"saving checkpoint to {checkpoint_out_dir}")
        torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
        
        # Force garbage collection and clear cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()

def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank, running
    global epoch, iter_num, raw_model, optimizer, encoder_args, decoder_args, checkpoint_out_dir
    # Register signal handler for keyboard interrupts
    signal.signal(signal.SIGINT, signal_handler)

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/VQ')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    print('prepare dataloader...')
    files = Path(args.dataset_dir).rglob('*.pkl')
    files = [file for file in files]
    # tuh = Path("/workspace/tuh_full/train").rglob('*.pkl')
    # files += [file for file in tuh]
    dataset_train = PickleLoader(files)
    files = Path(args.dataset_dir+"_test").rglob('*.pkl')
    files = [file for file in files]
    dataset_test = PickleLoader(files)
    print('finished!')

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
    num_workers=6,  # Reduce from 10
    pin_memory=False,  # Change to False
    drop_last=True,
    shuffle=True,
    prefetch_factor=2  # Add this to reduce prefetching
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size*2,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )
    print(dtype)
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0

    # model init
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=128)



    # Check if there are any checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_out_dir) if f.startswith('ckpt-') and f.endswith('.pt')]
    
    if checkpoint_files:
        # Extract the checkpoint numbers and find the highest one
        checkpoint_nums = []
        for f in checkpoint_files:
            try:
                num = int(f.split('-')[1].split('.')[0])
                checkpoint_nums.append(num)
            except (ValueError, IndexError):
                continue
        
        if checkpoint_nums:
            highest_checkpoint = max(checkpoint_nums)
            ckpt_path = os.path.join(checkpoint_out_dir, f'ckpt-{highest_checkpoint}.pt')
            print(f"Found checkpoint: {ckpt_path}")
            init_from = 'resume'
        else:
            init_from = 'scratch'
    else:
        init_from = 'scratch'
        
    try:
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            encoder_conf = NTConfig(**encoder_args)
            decoder_conf = NTConfig(**decoder_args)
            model = VQ_Align(encoder_conf, decoder_conf, "/workspace/VQ.pt", periodic_decoder_config={"hidden_dim":384})
            start_epoch = 0
        elif init_from == 'resume':
            print(f"Resuming training from {checkpoint_out_dir}")
            # resume training from a checkpoint.
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            # create the model
            encoder_conf = NTConfig(**encoder_args)
            decoder_conf = NTConfig(**decoder_args)
            model = VQ_Align(encoder_conf, decoder_conf,ckpt_path, periodic_decoder_config={"hidden_dim":384})
            
            # Load state dict
            iter_num = checkpoint['iter_num']
            start_epoch = checkpoint['epoch'] + 1

        model.to(device).to(ptdtype)
        import torch.nn.functional as F

        # model.VQ.loss_fn = F.smooth_l1_loss

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
        if init_from == 'resume':
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("Didnt load optimizer state")
        checkpoint = None # free up memory

        # compile the model
        if args.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0
                        # Add this in model initialization
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()

        # wrap model into DDP container
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
        lr_schedule_values = cosine_scheduler(
            args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs
        )
        lr_schedule_values_delayed = cosine_scheduler(
                    args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
                    warmup_epochs=args.warmup_epochs
                )

        # training loop
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = model.module if ddp else model # unwrap DDP container if needed
        losses = {}
        iter_num = local_iter_num#iter_num%(num_training_steps_per_epoch*args.epochs)
        # model.VQ.force_update_alpha(-3.0)
        for epoch in range(start_epoch, 100000):
            if not running:
                break
                
            try:
                for step, (batch) in enumerate(data_loader_train):
                    if not running:
                        break
                        
                    # determine and set the learning rate for this iteration
                    if iter_num >= len(lr_schedule_values):
                        raise Exception("training finished")
                        
                    lr_fast = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
                    lr_slow = lr_schedule_values_delayed[iter_num] if args.decay_lr else args.learning_rate
                    for lr,ii,param_group in zip([lr_slow,lr_slow,lr_fast,lr_fast,lr_fast], [1.,1., 1., 1., 200.], optimizer.param_groups):
                        param_group['lr'] = lr*ii
                        if ii == 1000 and step == 0 and epoch == start_epoch:
                            print("domain classifier lr: ", param_group['lr'], "domain classifier weight decay: ", param_group['weight_decay'])

                    # forward backward update, with optional gradient accumulation to simulate larger batch size
                    # and using the GradScaler if data type is float16
                    if ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
                    
                    X, Y_freq, Y_raw, input_chans, input_time, input_mask, target = batch
                    X = X.float().to(device, non_blocking=True).to(ptdtype)
                    Y_freq = Y_freq.float().to(device, non_blocking=True).to(ptdtype)
                    Y_raw = Y_raw.float().to(device, non_blocking=True).to(ptdtype)
                    input_chans = input_chans.to(device, non_blocking=True)
                    input_time = input_time.to(device, non_blocking=True)
                    input_mask = input_mask.to(device, non_blocking=True)


                    ################################
                    if np.random.random()>0.5:
                        kernel_size, sigma = 30, 6
                        noise = torch.randn_like(X)
                        # Create a 1D Gaussian kernel
                        kernel = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1)**2 / (2*sigma**2))**0.1
                        kernel = kernel / kernel.sum()
                        kernel = kernel.view(1, 1, -1).type_as(noise)
                        while len(kernel.shape) < len(noise.shape):
                            kernel = kernel.unsqueeze(0)
                        noise = F.conv1d(
                            noise.reshape(-1, 1, noise.shape[-1]),  # Reshape for conv1d
                            kernel.to(noise.device),
                            padding='same'
                        ).view(noise.shape).to(X.device).to(X.dtype)
                        abs_scale = X.std(dim=-1, keepdim=True)
                        noise_scale_percent =  np.random.random()**2/3
                        X = X*(1-noise_scale_percent)+ noise*(abs_scale)*noise_scale_percent
                    
                    with ctx:
                        loss, _, log = model(X, Y_freq, Y_raw, input_chans, input_time, input_mask, target=target)
                        for k,v in log.items():
                            if k not in losses:
                                losses[k] = []
                            losses[k].append(v)
                        if loss != loss:
                            continue
                            print("nan")
                        #domain_loss2 = model(X_text)
                        loss = (loss) / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    # backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # clip the gradient
                        if args.grad_clip != 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        # print("Alpha value: ", model.VQ.sigmodule_alpha.data, "Alpha grad: ", model.VQ.sigmodule_alpha.grad.data)
                        # step the optimizer and scaler if training in fp16
                        scaler.step(optimizer)
                        scaler.update()

                        # model.VQ.forc
                        # force_update_alpha((model.VQ.sigmodule_alpha.data.float()*0.99).item())
                        optimizer.zero_grad(set_to_none=True)
                        
                    # evaluate the loss on train/val sets and wxrite checkpoints
                    if (iter_num + 1) % args.log_interval == 0 and master_process and running:
                        M = lambda k: sum(losses[k])/len(losses[k])
                        with torch.no_grad():
                            _alpha = f"{model.VQ.sigmodule_alpha.data}"
                        print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train loss {M('train/total_loss'):.4f}, freq loss {M('train/rec_freq_loss'):.4f}, raw loss {M('train/rec_raw_loss'):.4f} |        lr: {lr:.7f}   [alpha={_alpha}]")
                        print(f"Domain loss: {M('train/domain_loss'):.4f}, Accuracy: {M('train/accuracy'):.4f}, Contrastive loss: {M('train/contrastive_loss'):.4f}, Similarity matrix: {M('train/similarity_matrix'):.4f}")
                        losses = {}

                    # timing and logging
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                
                    iter_num += 1
                    local_iter_num += 1
                
                # Run validation at the end of each epoch
                if running:
                    print(f"Running validation for epoch {epoch}")
                    val_losses = {}
                    model.eval()
                    try:
                        with torch.no_grad():
                            for val_step, (val_batch) in enumerate(data_loader_test):
                                X, Y_freq, Y_raw, input_chans, input_time, input_mask, target = val_batch
                                X = X.float().to(device, non_blocking=True).to(ptdtype)
                                Y_freq = Y_freq.float().to(device, non_blocking=True).to(ptdtype)
                                Y_raw = Y_raw.float().to(device, non_blocking=True).to(ptdtype)
                                input_chans = input_chans.to(device, non_blocking=True)
                                input_time = input_time.to(device, non_blocking=True)
                                input_mask = input_mask.to(device, non_blocking=True)
                                
                                try:
                                    val_loss, _, val_log = model(X, Y_freq, Y_raw, input_chans, input_time, input_mask, target=target)
                                    
                                    for k, v in val_log.items():
                                        k = k.replace('train/', 'val/')  # Change prefix from train to val
                                        if k not in val_losses:
                                            val_losses[k] = []
                                        val_losses[k].append(v)
                                except RuntimeError as e:
                                    if "CUDA error: an illegal memory access was encountered" in str(e):
                                        print(f"CUDA memory error during validation batch {val_step}, skipping this batch")
                                        torch.cuda.empty_cache()
                                        continue
                                    else:
                                        raise e
                        
                            # Print validation results if we have any data
                            if val_losses:
                                M_val = lambda k: sum(val_losses[k])/len(val_losses[k])
                                print(f"Validation epoch {epoch}: val loss {M_val('val/total_loss'):.4f}, freq loss {M_val('val/rec_freq_loss'):.4f}, raw loss {M_val('val/rec_raw_loss'):.4f}")
                                print(f"Val Domain loss: {M_val('val/domain_loss'):.4f}, Accuracy: {M_val('val/accuracy'):.4f}, Contrastive loss: {M_val('val/contrastive_loss'):.4f}, Similarity matrix: {M_val('val/similarity_matrix'):.4f}")
                            else:
                                print(f"No valid batches processed during validation for epoch {epoch}")
                    except Exception as e:
                        print(f"Error during validation: {e}")
                        # Clear CUDA cache to potentially recover
                        torch.cuda.empty_cache()
                        
                        model.train()
            except Exception as e:
                if running:
                    print(f"Exception occurred: {e}")
                raise e
                continue
            if (epoch + 1) % args.save_ckpt_freq == 0 and master_process and running:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'encoder_args': encoder_args,
                    'decoder_args': decoder_args,
                    'iter_num': iter_num,
                    'epoch': epoch
                }, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

    except Exception as e:
        print(f"Exception occurred: {e}")
        print(f"iter_num: {iter_num}, epoch: {epoch}, local_iter_num: {local_iter_num}")
        
        # Save checkpoint on exception
        if local_iter_num>10:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'encoder_args': encoder_args,
                    'decoder_args': decoder_args,
                    'iter_num': iter_num,
                    'epoch': epoch
                }, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))


        raise e

    if ddp:
        destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--log_interval', default=15, type=int)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=5*2, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=3, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=7e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--min_lr', type=float, default=2e-6)
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=True, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dtype = 'float32' #'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    
    main(args)

