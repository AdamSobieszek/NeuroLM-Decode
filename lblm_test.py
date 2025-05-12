# File: test_lblm_model.py

import torch
import unittest

try:
    from model2 import InnerSpeech_LBLM_MSTP # Assuming model2 is in PYTHONPATH or ./model2
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'model2' package is correctly set up and in PYTHONPATH.")
    print("Expected structure: ./model2/[files].py, with this script in the parent directory.")
    print("Make sure model2/__init__.py exists and correctly exports InnerSpeech_LBLM_MSTP.")
    raise

def get_default_model_params(
    eeg_channels=3,
    conformer_num_layers=2,
    conformer_n_head=2,
    timepoints=50, # Default timepoints for calculating num_patches
    patch_length=25,
    patch_stride=6,
    num_subjects=5,
    subject_embed_dim=1,
    mask_ratio=0.15,
    use_rev_in=True,
    conformer_input_dim=32,
    conformer_d_model=32,
    conformer_d_ff=64,
    conformer_dropout_rate=0.0,
    conformer_conv_kernel_size=7,
    huber_delta=1.0,
    lambda_amplitude=0.1,
    lambda_phase=0.1
    ):
    """Helper to get default parameters for testing, allowing overrides."""
    # Max patches in LBLMInputProcessor needs to be sufficient for the given timepoints
    # Current LBLMInputProcessor calculates its own max_patches based on a hardcoded 500 timepoints.
    # This should be fine unless timepoints in tests become extremely large.
    # For timepoints=timepoints: num_patches = (timepoints - patch_length) // patch_stride + 1

    return {
        "patch_length": patch_length,
        "patch_stride": patch_stride,
        "num_subjects": num_subjects,
        "subject_embed_dim": subject_embed_dim,
        "mask_ratio": mask_ratio,
        "use_rev_in": use_rev_in,
        "eeg_channels": eeg_channels,

        "conformer_input_dim": conformer_input_dim,
        "conformer_num_layers": conformer_num_layers,
        "conformer_d_model": conformer_d_model,
        "conformer_n_head": conformer_n_head,
        "conformer_d_ff": conformer_d_ff,
        "conformer_dropout_rate": conformer_dropout_rate,
        "conformer_conv_kernel_size": conformer_conv_kernel_size,

        "huber_delta": huber_delta,
        "lambda_amplitude": lambda_amplitude,
        "lambda_phase": lambda_phase
    }

class TestLBLMModel(unittest.TestCase):

    def assertTensorsFinite(self, tensor_dict):
        for name, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor):
                self.assertTrue(torch.isfinite(tensor).all(), f"Tensor {name} contains non-finite values (NaN or Inf)")

    def _run_inference_test(self, model_params_dict, batch_size, timepoints_for_data, device,
                              subject_ids_provided=True, input_mask_override=None, test_name=""):
        print(f"\n--- Running Test: {test_name} ---")
        print(f"Params: BS={batch_size}, Timepoints={timepoints_for_data}, Device={device}, Subjects={subject_ids_provided}")
        
        model = InnerSpeech_LBLM_MSTP(**model_params_dict).to(device)
        model.eval()

        num_channels = model_params_dict["eeg_channels"]
        dummy_eeg = torch.randn(batch_size, num_channels, timepoints_for_data).to(device)

        dummy_subject_ids = None
        if subject_ids_provided and model_params_dict["num_subjects"] > 0:
            dummy_subject_ids = torch.randint(0, model_params_dict["num_subjects"], (batch_size,)).to(device)
        elif subject_ids_provided and model_params_dict["num_subjects"] == 0:
            print("Warning: subject_ids_provided=True but model_params['num_subjects']=0. No subject IDs will be used.")

        with torch.no_grad():
            try:
                loss, _, log_output = model(dummy_eeg, subject_ids=dummy_subject_ids, input_mask_override=input_mask_override)
            except Exception as e:
                self.fail(f"Model forward pass failed for {test_name} with error: {e}\n{traceback.format_exc()}")

        print(f"Loss ({test_name}): {loss.item()}")
        print(f"Log ({test_name}): {log_output}")

        self.assertIsInstance(loss, torch.Tensor, f"Loss is not a tensor for {test_name}")
        self.assertTrue(torch.isfinite(loss).all(), f"Loss is not finite for {test_name}")
        self.assertIsInstance(log_output, dict, f"Log output is not a dict for {test_name}")
        self.assertTensorsFinite(log_output)

        model.train()
        optimizer = model.configure_optimizers(0.01, 1e-4, (0.9,0.999), device.type)
        optimizer.zero_grad()
        loss_train, _, log_train = model(dummy_eeg, subject_ids=dummy_subject_ids, input_mask_override=input_mask_override)
        
        # Check if any tokens were actually masked, if not, loss might be 0 and no grad
        total_masked_tokens = 0
        if input_mask_override is not None:
            total_masked_tokens = input_mask_override.sum().item()
        elif 'masked_indices_bool' in model.input_processor.__dict__: # Heuristic to check if internal mask exists
             # This part is tricky without direct access to the internal mask from the test.
             # We rely on mask_ratio > 0 for internal masking.
            if model_params_dict.get("mask_ratio", 0) > 0:
                 # Can't know for sure without inspecting forward pass, assume some were masked
                 pass # Assume some masking if ratio > 0

        if loss_train.requires_grad:
            try:
                loss_train.backward()
                # Basic check for any gradient presence if loss was non-zero
                if loss_train.item() > 1e-9: # If loss is effectively non-zero
                    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
                    self.assertTrue(has_grads, f"No gradients found after backward for {test_name} despite non-zero loss.")
                optimizer.step()
                print(f"Backward pass and optimizer step successful for {test_name}.")
            except Exception as e:
                import traceback
                self.fail(f"Model backward pass or optimizer step failed for {test_name} with error: {e}\n{traceback.format_exc()}")
        elif log_train['mstp/total_loss'].item() < 1e-9 : # If loss is zero (e.g. no masked tokens)
            print(f"Loss is zero for {test_name}. Skipping backward pass test as expected.")
        else: # Loss doesn't require grad but is non-zero, which is unexpected
             self.fail(f"Loss does not require grad for {test_name} but is non-zero. This is unexpected.")


    def test_basic_cpu(self):
        current_timepoints = 70
        params = get_default_model_params(eeg_channels=3, conformer_num_layers=1, conformer_n_head=2, timepoints=current_timepoints)
        self._run_inference_test(params, batch_size=2, timepoints_for_data=current_timepoints, device=torch.device("cpu"), test_name="Basic CPU")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_basic_gpu(self):
        current_timepoints = 70
        params = get_default_model_params(eeg_channels=3, conformer_num_layers=1, conformer_n_head=2, timepoints=current_timepoints)
        self._run_inference_test(params, batch_size=2, timepoints_for_data=current_timepoints, device=torch.device("cuda"), test_name="Basic GPU")

    def test_no_subject_ids(self):
        current_timepoints = 60
        # Test 1: Model configured with num_subjects=0
        params_no_subj_config = get_default_model_params(eeg_channels=2, num_subjects=0, timepoints=current_timepoints)
        self._run_inference_test(params_no_subj_config, batch_size=2, timepoints_for_data=current_timepoints, device=torch.device("cpu"),
                                 subject_ids_provided=False, test_name="No Subject IDs (num_subjects=0 in config)")

        # Test 2: Model configured with subjects, but subject_ids=None passed to forward
        params_with_subj_config = get_default_model_params(eeg_channels=2, num_subjects=5, timepoints=current_timepoints)
        self._run_inference_test(params_with_subj_config, batch_size=2, timepoints_for_data=current_timepoints, device=torch.device("cpu"),
                                 subject_ids_provided=False, test_name="No Subject IDs (subject_ids=None in forward)")


    def test_different_seq_len(self):
        # Shorter sequence (min timepoints = patch_length, e.g., 1 patch)
        tp_short = 25 # This will result in 1 patch
        params_short = get_default_model_params(timepoints=tp_short, patch_length=25, patch_stride=6)
        self._run_inference_test(params_short, batch_size=2, timepoints_for_data=tp_short, device=torch.device("cpu"), test_name="Short Sequence (1 patch)")

        # Longer sequence
        tp_long = 150
        params_long = get_default_model_params(timepoints=tp_long)
        self._run_inference_test(params_long, batch_size=2, timepoints_for_data=tp_long, device=torch.device("cpu"), test_name="Longer Sequence")

    def test_different_batch_size(self):
        current_timepoints = 50
        params = get_default_model_params(timepoints=current_timepoints)
        self._run_inference_test(params, batch_size=1, timepoints_for_data=current_timepoints, device=torch.device("cpu"), test_name="Batch Size 1")
        self._run_inference_test(params, batch_size=8, timepoints_for_data=current_timepoints, device=torch.device("cpu"), test_name="Batch Size 8")

    def test_mask_override_no_mask(self):
        current_timepoints = 50
        # Get params first to calculate num_patches correctly
        temp_params = get_default_model_params(mask_ratio=0.0, timepoints=current_timepoints) # mask_ratio in model is 0.0
        
        num_patches = (current_timepoints - temp_params["patch_length"]) // temp_params["patch_stride"] + 1
        batch_size = 2
        num_bm = batch_size * temp_params["eeg_channels"]
        override_mask = torch.zeros(num_bm, num_patches, dtype=torch.bool) # All False -> no tokens masked externally
        
        # The model's internal mask_ratio is 0.0, so it won't mask anything either.
        self._run_inference_test(temp_params, batch_size=batch_size, timepoints_for_data=current_timepoints, device=torch.device("cpu"),
                                 input_mask_override=override_mask, test_name="Mask Override (External No Mask, Internal No Mask)")

    def test_mask_override_all_masked(self):
        current_timepoints = 50
        # Get params first
        temp_params = get_default_model_params(timepoints=current_timepoints) # Internal mask_ratio is default (e.g., 0.15)
        
        num_patches = (current_timepoints - temp_params["patch_length"]) // temp_params["patch_stride"] + 1
        batch_size = 2
        num_bm = batch_size * temp_params["eeg_channels"]
        override_mask = torch.ones(num_bm, num_patches, dtype=torch.bool) # All True -> all tokens masked externally
        
        self._run_inference_test(temp_params, batch_size=batch_size, timepoints_for_data=current_timepoints, device=torch.device("cpu"),
                                 input_mask_override=override_mask, test_name="Mask Override (All Masked Externally)")

    def test_rev_in_disabled(self):
        current_timepoints = 60
        params = get_default_model_params(use_rev_in=False, timepoints=current_timepoints)
        self._run_inference_test(params, batch_size=2, timepoints_for_data=current_timepoints, device=torch.device("cpu"), test_name="RevIN Disabled")

import traceback # Add this at the top of the file

if __name__ == '__main__':
    print("Running LBLM Model Tests...")
    # Adding verbosity to unittest output
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner, argv=['first-arg-is-ignored'], exit=False)
    print("Finished LBLM Model Tests.")