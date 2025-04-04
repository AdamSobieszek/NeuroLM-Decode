
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

from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig
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
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    
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
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def save_checkpoint(raw_model, optimizer, encoder_args, decoder_args, iter_num, epoch, checkpoint_out_dir):
    if master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'encoder_args': encoder_args,
            'decoder_args': decoder_args,
            'iter_num': iter_num,
            'epoch': epoch
        }
        print(f"saving checkpoint to {checkpoint_out_dir}")
        torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank, running

    # Register signal handler for keyboard interrupts
    signal.signal(signal.SIGINT, signal_handler)

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/VQ')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    print('prepare dataloader...')
    files = Path(args.dataset_dir, 'train').rglob('*.pkl')
    files = [file for file in files]
    dataset_train = PickleLoader(files)
    print('finished!')

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
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

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0

    # model init
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=128)


    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'scratch'

    try:
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            encoder_conf = NTConfig(**encoder_args)
            decoder_conf = NTConfig(**decoder_args)
            model = VQ_Align(encoder_conf, decoder_conf, "/workspace/VQ.pt")
            start_epoch = 0
        elif init_from == 'resume':
            print(f"Resuming training from {checkpoint_out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(checkpoint_out_dir, 'VQ.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['encoder_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
                encoder_args[k] = checkpoint_model_args[k]
            checkpoint_model_args = checkpoint['decoder_args']
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
                decoder_args[k] = checkpoint_model_args[k]
            # create the model
            encoder_conf = NTConfig(**encoder_args)
            decoder_conf = NTConfig(**decoder_args)
            model = VQ_Align(encoder_conf, decoder_conf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            start_epoch = checkpoint['epoch'] + 1

        model.to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
        if init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory

        # compile the model
        if False:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0

        # wrap model into DDP container
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
        lr_schedule_values = cosine_scheduler(
            args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs
        )

        # training loop
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = model.module if ddp else model # unwrap DDP container if needed
        for epoch in range(start_epoch, args.epochs):
            if not running:
                break
                
            for step, (batch) in enumerate(data_loader_train):
                if not running:
                    break
                    
                print("step")
                # determine and set the learning rate for this iteration
                lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
                for ii,param_group in zip([1,0.8,2], optimizer.param_groups):
                    param_group['lr'] = lr*ii

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
                
                X, Y_freq, Y_raw, input_chans, input_time, input_mask = batch
                X = X.float().to(device, non_blocking=True)
                Y_freq = Y_freq.float().to(device, non_blocking=True)
                Y_raw = Y_raw.float().to(device, non_blocking=True)
                input_chans = input_chans.to(device, non_blocking=True)
                input_time = input_time.to(device, non_blocking=True)
                input_mask = input_mask.to(device, non_blocking=True)

                with ctx:
                    loss, _, log = model(X, Y_freq, Y_raw, input_chans, input_time, input_mask)
                    if loss != loss:
                        raise Exception
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
                    # step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)

                # evaluate the loss on train/val sets and write checkpoints
                if (iter_num + 1) % args.log_interval == 0 and master_process:
                    print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train loss {log['train/total_loss']:.4f}, freq loss {log['train/rec_freq_loss']:.4f}, raw loss {log['train/rec_raw_loss']:.4f}")
                    print(f"learning rate {lr}")

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                iter_num += 1
                local_iter_num += 1
            
            save_checkpoint(model, optimizer, encoder_args, decoder_args, iter_num, epoch, checkpoint_out_dir)
            
            if (epoch + 1) % args.save_ckpt_freq == 0 and master_process:
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
        # Save checkpoint on exception
        save_checkpoint(model, optimizer, encoder_args, decoder_args, iter_num, epoch, checkpoint_out_dir)
        raise  # Re-raise the exception after saving

    if ddp:
        destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--log_interval', default=30, type=int)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=18, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=5e-6, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--min_lr', type=float, default=1e8)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)

