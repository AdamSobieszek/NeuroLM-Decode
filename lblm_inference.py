from __future__ import annotations
import os, sys, argparse, json, random, types, collections
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from contextlib import nullcontext

# --------------------  try to import model -------------------------
try:
    from model2 import InnerSpeech_LBLM_MSTP
except ImportError as e:
    print(f"[inference] Could not import InnerSpeech_LBLM_MSTP: {e}")
    print(" Make sure the 'model2' package is visible in PYTHONPATH.")
    sys.exit(1)

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


TensorLike = Union[torch.Tensor, np.ndarray]
# ------------------------------------------------------------------
#            0.  helpers  –  reconstruct model_args, dtype …
# ------------------------------------------------------------------
def _recover_model_args(chk: Dict[str, Any]) -> Tuple[Dict[str, Any], types.SimpleNamespace | None]:
    """
    Return (model_args_dict, args_cli_namespace_or_None)
    """
    if 'model_args' in chk:
        model_args = chk['model_args']
    elif 'args_cli' in chk:
        # rebuild from saved cli args
        cli = chk['args_cli']
        model_args = {
            "patch_length": cli.patch_length,
            "patch_stride": cli.patch_stride,
            "num_subjects": cli.num_subjects,
            "subject_embed_dim": cli.subject_embed_dim,
            "mask_ratio": cli.mask_ratio,
            "use_rev_in": not getattr(cli, "no_rev_in", False),
            "eeg_channels": cli.eeg_channels,
            "conformer_input_dim": cli.conformer_embed_dim,
            "conformer_num_layers": cli.conformer_layers,
            "conformer_d_model": cli.conformer_d_model,
            "conformer_n_head": cli.conformer_n_head,
            "conformer_d_ff": cli.conformer_d_ff,
            "conformer_dropout_rate": cli.dropout,
            "conformer_conv_kernel_size": cli.conformer_conv_kernel,
            "huber_delta": cli.huber_delta,
            "lambda_amplitude": cli.lambda_amplitude,
            "lambda_phase": cli.lambda_phase,
        }
    else:
        raise ValueError("Checkpoint contains neither 'model_args' nor 'args_cli'")
    return model_args, chk.get('args_cli', None)


def _str_to_dtype(s: str | None, default=torch.float32) -> torch.dtype:
    if s is None:
        return default
    lookup = {"float32": torch.float32, "fp32": torch.float32,
              "float16": torch.float16, "fp16": torch.float16,
              "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    return lookup.get(str(s).lower(), default)


# ------------------------------------------------------------------
#            1.  Public  loader
# ------------------------------------------------------------------
def load_lblm_checkpoint(
        ckpt_path: str | os.PathLike,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        override_model_args: Optional[Dict[str, Any]] = None,
        silence: bool = False
) -> Tuple[InnerSpeech_LBLM_MSTP, Dict[str, Any], torch.dtype]:
    """
    Returns (model, model_args, dtype)
    """
    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(ckpt_path)
    device = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    chk = torch.load(ckpt_path, map_location=device)

    model_args, saved_cli = _recover_model_args(chk)
    if override_model_args:
        model_args.update(override_model_args)

    model = InnerSpeech_LBLM_MSTP(**model_args)

    # strip DDP 'module.' prefix if required
    state = chk["model"]
    if any(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    # tolerate missing keys
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not silence and (missing or unexpected):
        print(f"[load] state_dict loaded – missing={len(missing)}, unexpected={len(unexpected)}")

    # dtype logic
    if dtype is not None:
        dtype = _str_to_dtype(dtype)
    else:
        dtype = _str_to_dtype(getattr(saved_cli, "dtype", None))

    model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    model.eval()
    if not silence:
        pcount = sum(p.numel() for p in model.parameters())
        print(f"[load] Model ready ({pcount:,} params) – device={device} dtype={dtype}")
    return model, model_args, dtype


# ==================================================================
#                    2.  High-level Engine
# ==================================================================
class LBLMInferenceEngine:
    """
    One object to rule them all:
        >>> engine = LBLMInferenceEngine("out/ckpt_final.pt", device="cuda")
        >>> preds  = engine.predict(eeg, subject_id=2)
        >>> recon  = engine.reconstruct_patches(eeg, subject_id=2, strategy='block')
    """

    def __init__(self,
                 ckpt_path: str | os.PathLike,
                 device: str | torch.device | None = None,
                 dtype: str | torch.dtype | None = None,
                 override_model_args: Optional[Dict[str, Any]] = None,
                 silence: bool = False):
        self.model, self.model_args, self.dtype = load_lblm_checkpoint(
            ckpt_path, device, dtype, override_model_args, silence=silence
        )
        self.device = next(self.model.parameters()).device
        self._autocast = (torch.autocast(device_type="cuda", dtype=self.dtype)
                          if self.device.type == "cuda" and self.dtype != torch.float32
                          else nullcontext())

    # --------------------------- utils -----------------------------
    @staticmethod
    def _prepare_inputs(x: TensorLike | Sequence[TensorLike],
                        subj: int | Sequence[int],
                        device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # EEG tensor
        if isinstance(x, (list, tuple)):
            x = [torch.as_tensor(xx, dtype=torch.float32) for xx in x]
            x = torch.stack(x, 0)
        else:
            x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim == 2:     # (C,T)
                x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError("EEG must have shape (C,T) or (B,C,T)")
        # subject
        if isinstance(subj, (list, tuple, np.ndarray)):
            subj = torch.as_tensor(subj, dtype=torch.long)
        else:
            subj = torch.as_tensor([int(subj)], dtype=torch.long)
        if subj.ndim == 0:
            subj = subj.unsqueeze(0)
        # move
        x = x.to(device, non_blocking=True)
        subj = subj.to(device, non_blocking=True)
        if dtype != torch.float32:
            x = x.to(dtype)
        return x, subj

    # --------------------------- public ----------------------------
    def predict(self,
                x_eeg: TensorLike | Sequence[TensorLike],
                subject_id: int | Sequence[int] = 0,
                mask_ratio: Optional[float] = None,
                return_numpy: bool = False,
                **forward_kwargs) -> Dict[str, Any]:
        """
        Simple forward pass – returns whatever the model returns
        (converted to numpy if return_numpy=True).
        """
        x, subj = self._prepare_inputs(x_eeg, subject_id, self.device, self.dtype)

        original_ratio = None
        if mask_ratio is not None:
            original_ratio = getattr(self.model, "mask_ratio", None)
            self.model.mask_ratio = mask_ratio

        with torch.no_grad(), self._autocast:
            out = self.model(x, subject_ids=subj, **forward_kwargs)

        # restore original mask ratio
        if original_ratio is not None:
            self.model.mask_ratio = original_ratio

        # normalise output into dict
        if len(out) == 3:
            return out[1]
        elif isinstance(out, dict):
            preds = out
        elif isinstance(out, (tuple, list)):
            # common layout: loss, pred_wave, pred_amp, pred_phase
            if len(out) >= 4 and torch.is_tensor(out[1]):
                _, w, a, p = out[:4]
                preds = {"wave": w, "amplitude": a, "phase": p}
            elif len(out) >= 2 and isinstance(out[1], dict):
                preds = out[1]
            else:
                preds = {"raw": out}
        else:
            preds = {"raw": out}

        if return_numpy:
            preds = {k: (v.detach().cpu().float().numpy() if torch.is_tensor(v) else v)
                     for k, v in preds.items()}
        return preds

    # ------------------------------------------------------------------
    #                     Advanced  reconstruction
    # ------------------------------------------------------------------
    def reconstruct_patches(
            self,
            x_eeg: TensorLike,
            subject_id: int = 0,
            strategy: str = 'random',
            block_settings: Optional[Dict[str, Any]] = None,
            custom_mask: Optional[torch.Tensor] = None,
            return_numpy: bool = True
    ) -> Dict[str, Any]:
        """
        Reconstructs *masked patches* according to `strategy`.
        Currently supports only batch size == 1.
        """
        strategy = strategy.lower()
        x, subj = self._prepare_inputs(x_eeg, subject_id, self.device, self.dtype)
        if x.shape[0] != 1:
            raise NotImplementedError("reconstruct_patches currently supports batch size 1")

        with torch.no_grad(), self._autocast:
            emb, orig, internal_mask_flat, rev_mean, rev_std = \
                self.model.input_processor(x, subj)

        # reshape helpers
        num_ch = self.model_args["eeg_channels"]
        num_patches = emb.shape[1]
        internal_mask = internal_mask_flat.reshape(num_ch, num_patches)

        # decide final mask
        if strategy == 'random':
            final_mask = internal_mask
        elif strategy == 'none':
            final_mask = torch.zeros_like(internal_mask, dtype=torch.bool)
        elif strategy == 'block':
            if block_settings is None:
                block_settings = {'channel_idx': 0, 'size': 5, 'start': 'random'}
            c = block_settings.get('channel_idx', 0)
            size = block_settings.get('size', 5)
            start = block_settings.get('start', 'random')
            final_mask = torch.zeros_like(internal_mask, dtype=torch.bool)
            if num_patches < 1:
                pass
            elif num_patches <= size:
                final_mask[c, :] = True
            else:
                s = random.randint(0, num_patches - size) if start == 'random' else int(start)
                final_mask[c, s:s+size] = True
        elif strategy == 'custom':
            if custom_mask is None:
                raise ValueError("custom_mask tensor must be provided for strategy='custom'")
            if custom_mask.shape != (num_ch, num_patches):
                raise ValueError(f"custom_mask must have shape {(num_ch, num_patches)}")
            final_mask = custom_mask.to(self.device, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        # build conformer input: zero-out masked patches
        conf_in = emb.clone()
        conf_in[final_mask] = 0.0
        backbone = self.model.conformer_backbone(conf_in, attention_mask=final_mask)
        masked_backbone = backbone[final_mask]      # [N_masked, d_model]

        # nothing to predict?
        if masked_backbone.numel() == 0:
            return {"pred_wave": np.empty((0, self.model.patch_length)),
                    "pred_amplitude": np.empty((0, self.model.patch_length//2+1)),
                    "pred_phase": np.empty((0, self.model.patch_length//2+1)),
                    "applied_mask": final_mask.cpu().numpy()}

        pred_wave, pred_amp, pred_phase = self.model.prediction_heads(masked_backbone)

        # reverse RevIN
        if self.model.input_processor.use_rev_in and rev_mean is not None:
            m_sel = rev_mean[0][final_mask]   # [N_masked,1]
            s_sel = rev_std [0][final_mask]
            pred_wave = self.model.input_processor.rev_in_denorm(pred_wave, m_sel, s_sel)

        orig_patches = orig[0]                      # [C, N_patches, patch_len]
        orig_masked  = orig_patches[final_mask]

        to_np = lambda t: t.detach().cpu().float().numpy() if return_numpy else t.detach()

        return {
            "pred_wave":      to_np(pred_wave),
            "pred_amplitude": to_np(pred_amp),
            "pred_phase":     to_np(pred_phase),
            "original_masked_wave": to_np(orig_masked),
            "applied_mask":   to_np(final_mask),
            "patch_len":      self.model.patch_length
        }


# ==================================================================
#                    3.  Command-line interface
# ==================================================================
def _cli_predict(p: argparse.ArgumentParser):
    p.add_argument("--eeg", required=True,
                   help="Path to .npy file containing EEG (channels,time)")
    p.add_argument("--subject", default=0, type=int)
    p.add_argument("--save_json", default=None,
                   help="Save predictions to JSON file")
    p.add_argument("--mask_ratio", default=None, type=float,
                   help="Override model.mask_ratio during this call")
    return p


def _cli_reconstruct(p: argparse.ArgumentParser):
    p.add_argument("--eeg", required=True)
    p.add_argument("--subject", default=0, type=int)
    p.add_argument("--strategy", default="random",
                   choices=["random", "none", "block", "custom"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save_json", default=None)
    p.add_argument("--block_channel", default=0, type=int)
    p.add_argument("--block_size", default=5, type=int)
    return p


    if args.cmd == "predict":
        preds = eng.predict(eeg, subject_id=args.subject, mask_ratio=args.mask_ratio)
        print("Prediction keys:", preds.keys())
        if args.save_json:
            with open(args.save_json, "w") as fp:
                json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in preds.items()}, fp)
    elif args.cmd == "reconstruct":
        if args.strategy == "block":
            block_settings = {"channel_idx": args.block_channel,
                              "size": args.block_size, "start": "random"}
        else:
            block_settings = None
        res = eng.reconstruct_patches(
            eeg, subject_id=args.subject, strategy=args.strategy,
            block_settings=block_settings
        )
        print(f"Reconstructed {res['pred_wave'].shape[0]} patches.")
        if args.save_json:
            with open(args.save_json, "w") as fp:
                json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in res.items() if k != "applied_mask"}, fp)
        if args.plot and _HAS_PLT and res['pred_wave'].shape[0] > 0:
            n = min(5, res['pred_wave'].shape[0])
            fig, axs = plt.subplots(n, 1, figsize=(10, 2.5*n), sharex=True)
            for i in range(n):
                axs[i].plot(res['original_masked_wave'][i], label="orig")
                axs[i].plot(res['pred_wave'][i], label="pred", ls="--")
                axs[i].legend()
            axs[-1].set_xlabel("sample")
            plt.tight_layout()
            plt.show()
        elif args.plot and not _HAS_PLT:
            print("matplotlib not available – cannot plot.")

