import os
import torch
import torch_npu

try:
    from .moe_token_permute_grad import MoeTokenPermuteGrad as _OrigMoeTokenPermuteGrad
except ImportError:
    from moe_token_permute_grad import MoeTokenPermuteGrad as _OrigMoeTokenPermuteGrad


class MoeTokenPermuteGrad(_OrigMoeTokenPermuteGrad):
    def __call__(self, permuted_output_grad, sorted_indices):
        test_mode = os.environ.get("PERM_GRAD_TEST", "")

        if test_mode == "clone_input":
            permuted_output_grad = permuted_output_grad.clone()
        elif test_mode == "fixed_indices":
            E_local = permuted_output_grad.shape[0]
            sorted_indices = torch.arange(E_local, dtype=torch.int32, device=permuted_output_grad.device)
        elif test_mode == "sync_before":
            torch_npu.npu.synchronize()
        elif test_mode == "warm_up":
            _ = super().__call__(permuted_output_grad, sorted_indices)
            torch_npu.npu.synchronize()

        return super().__call__(permuted_output_grad, sorted_indices)
