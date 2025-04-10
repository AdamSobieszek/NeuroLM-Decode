
Skip to Main
model_vq.py
Last Checkpoint: 53 minutes ago
Selection deleted
349
350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384
385
386
387
388
389
390
391
392
393
394
395
396
397
398
399
400
401
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419
420
421
422
423
424
425
426
427
428
429
430
431
432
433
434
435
436
437
438
class VQ_Align(nn.Module):
    def __init__(self, 
                 encoder_config=None,
                 decoder_config=None,
                 checkpoint_path=None
                 ):
        super(VQ_Align, self).__init__()
        if checkpoint_path:
            print("LOADING VQ.pt CHECKPOINT\n\n\n\n-----------------")
            self.VQ = load_model(checkpoint_path, "cuda")
        else:
            self.VQ = VQ(encoder_config, decoder_config)
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, y_freq=None, y_raw=None, input_chans=None, input_time=None, input_mask=None, return_reconstruction=False):
        if y_freq is not None:
            loss, encoder_features, log = self.VQ(x, y_freq, y_raw, input_chans, input_time, input_mask, return_reconstruction=False)
            # reverse_x = ReverseLayerF.apply(encoder_features, alpha)
            # domain_out = self.domain_classifier(reverse_x)
            # target = torch.full((domain_out.size(0), domain_out.size(1)), fill_value=-1, device=x.device)
            # target[input_mask == True] = 0
            # domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), target.view(-1), ignore_index=-1)
            split="train" if self.training else "val"
            # log[f'{split}/domain_loss'] = domain_loss.detach().item()
            return loss, None, log

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Identify AE parameters more precisely - look for specific parameter names
        # that belong to the encoder-decoder interface layers
        ae_params = []
        sigmodule_params = []
        decay_params = []
        nodecay_params = []
        
        # Categorize parameters by name and dimension
        for name, p in param_dict.items():
            # Parameters for encoder output layer and decoder input layer
            if 'cast_' in name:
                ae_params.append(p)
            elif 'sigmodule' in name:
                sigmodule_params.append(p)
            # Standard weight decay for 2D+ parameters that aren't in the AE interfaces
            elif p.dim() >= 2:
                decay_params.append(p)
            # No decay for 1D parameters (biases, etc.)
            else:
                nodecay_params.append(p)
        
        # Configure optimization groups with appropriate learning rates and weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay/2, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': ae_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': sigmodule_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            ]
            
        # Print parameter counts for debugging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_ae_params = sum(p.numel() for p in ae_params)
        num_sigmodule_params = sum(p.numel() for p in sigmodule_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"Num AE interface parameters: {len(ae_params)}, with {num_ae_params:,} parameters")
        print(f"Num sigmodule parameters: {len(sigmodule_params)}, with {num_sigmodule_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer

    

or 





