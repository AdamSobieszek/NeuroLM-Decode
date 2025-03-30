"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import torch
from torch import nn
import torch.nn.functional as F
import inspect

from model.model_neural_transformer import NeuralTransformer
from model.norm_ema_quantizer import NormEMAVectorQuantizer
 
from torch.autograd import Function

from model.model_neural_transformer import NTConfig

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class VQ(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=128,
                 latent_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=False,
                 decoder_out_dim=200,
                 smooth_l1_loss = False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config.in_chans != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config.in_chans} to {embed_dim}")
            decoder_config.in_chans = embed_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder_freq = NeuralTransformer(decoder_config)
        self.decoder_raw = NeuralTransformer(decoder_config)
                
        # self.quantize = NormEMAVectorQuantizer(
        #     n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        # )

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config.n_embd, encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(encoder_config.n_embd, embed_dim) # for quantize
        )
        self.decode_task_layer_freq = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim // 2),
        )
        self.decode_task_layer_raw = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim),
        )

        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer_freq.apply(self._init_weights)
        self.decode_task_layer_raw.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.embed_dim = embed_dim
        self.n_embd = encoder_config.n_embd
        self.latent_dim = latent_dim
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'quantize.embedding.weight', 'decoder.pos_embed', 'decoder.time_embed', 
    #             'encoder.pos_embed', 'encoder.time_embed'}
    def init_ae_layer(self):
        # Enhanced downcast layer with BatchNorm and GELU activations
        self.downcast_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.latent_dim),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
        )
        self.downcast_layer.apply(self._init_weights)
        
        # Symmetrical upcast layer for balanced upscaling
        self.upcast_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.latent_dim * 2, self.embed_dim),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
        )
        self.upcast_layer.apply(self._init_weights)
             
        
    @property
    def device(self):
        return self.decoder.cls_token.device
    
    def get_number_of_tokens(self):
        return self.quantize.n_e

    # def get_tokens(self, data, input_chans=None, input_times=None, mask=None, **kwargs):
    #     quantize, _, loss, _ = self.encode(data, input_chans, input_times, mask)
    #     return embed_ind.view(data.size(0), data.size(1))

    def encode(self, x, input_chans=None, input_time=None, mask=None):
        batch_size, n, t = x.shape
        encoder_features = self.encoder(x, input_chans, input_time, mask, return_all_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))
        print(to_quantizer_features.shape)
        quantize = self.downcast_layer(to_quantizer_features)
        print(quantize.shape)
        return quantize, None, None, encoder_features
        
    def decode(self, quantize, input_chans=None, input_time=None, mask=None, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        print(quantize.shape)
        quantize = self.upcast_layer(quantize)
        print(quantize.shape)
        decoder_features_freq = self.decoder_freq(quantize, input_chans, input_time, mask, return_all_tokens=True)
        decoder_features_raw = self.decoder_raw(quantize, input_chans, input_time, mask, return_all_tokens=True)
        rec_freq = self.decode_task_layer_freq(decoder_features_freq)
        rec_raw = self.decode_task_layer_raw(decoder_features_raw)
        
        return rec_freq, rec_raw
    
    # def get_codebook_indices(self, x, input_chans=None, input_time=None, input_mask=None, **kwargs):
    #     if input_mask is None:
    #         mask = None
    #     else:
    #         mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    #     return self.get_tokens(x, input_chans, input_time, mask, **kwargs)
    
    def calculate_rec_loss(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def forward(self, x, y_freq, y_raw, input_chans=None, input_time=None, input_mask=None, return_reconstruction=False, **kwargs):
        """
        x: shape [B, N, T]
        """
        mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        quantize, _, _, encoder_features = self.encode(x, input_chans, input_time, mask)
        
        xrec_freq, xrec_raw = self.decode(quantize, input_chans, input_time, mask)
        loss_freq_mask = input_mask.unsqueeze(-1).repeat(1, 1, xrec_freq.size(-1))
        loss_raw_mask = input_mask.unsqueeze(-1).repeat(1, 1, xrec_raw.size(-1))
        rec_freq_loss = self.calculate_rec_loss(xrec_freq * loss_freq_mask, y_freq)
        rec_raw_loss = self.calculate_rec_loss(xrec_raw * loss_raw_mask, y_raw)
        loss = rec_freq_loss + rec_raw_loss
        log = {}
        split="train" if self.training else "val"
        log[f'{split}/rec_freq_loss'] = rec_freq_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        if not return_reconstruction:
            return loss, encoder_features, log
        else:
            return (xrec_freq, xrec_raw), encoder_features,log
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
        
def load_model(ckpt_path, device):
    """
    Load the VQ_Align model from checkpoint
    """
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get model configuration
    encoder_args = checkpoint['encoder_args']
    decoder_args = checkpoint['decoder_args']
    
    if "dropout" in encoder_args:
        encoder_args["dropout"] = 0.1
        decoder_args["dropout"] = 0.1
    # Create model configuration
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    encoder_conf

    
    # Initialize model
    model = VQ_Align(encoder_conf, decoder_conf)
    
    # Fix state dict keys if needed
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    state_dict = {k:v for k,v in state_dict.items() if k not in ["domain_classifier.0.weight", "domain_classifier.0.bias", "domain_classifier.2.weight", "domain_classifier.2.bias", "wte.weight", "VQ.quantize.cluster_size", "VQ.quantize.embedding.weight", "VQ.quantize.embedding.cluster_size", "VQ.quantize.embedding.embed_avg", "VQ.quantize.embedding.initted"]}
    
    # Load state dict
    model.load_state_dict(state_dict)
    model = model.VQ
    model.init_ae_layer()
    # print(model)
    return model

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
        
        # self.domain_classifier = nn.Sequential(
        #         nn.Linear(decoder_config.n_embd, 256),
        #         nn.GELU(),
        #         nn.Linear(256, 2)
        #     )
        

        # model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        # sd_hf = model_hf.state_dict()
        # self.wte = nn.Embedding(50257, 768, _freeze=True)
        # self.wte.weight.data = sd_hf['transformer.wte.weight']
        
        # self.domain_classifier.apply(self._init_weights)
    
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
        decay_params = []
        nodecay_params = []
        
        # Categorize parameters by name and dimension
        for name, p in param_dict.items():
            # Parameters for encoder output layer and decoder input layer
            if 'encode_task_layer' in name or 'invert_encode_task_layer' in name:
                ae_params.append(p)
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
            # Higher learning rate but moderate weight decay for AE interface layers
            # to allow faster adaptation without causing instability
            {'params': ae_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            ]
            
        # Print parameter counts for debugging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_ae_params = sum(p.numel() for p in ae_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"Num AE interface parameters: {len(ae_params)}, with {num_ae_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer