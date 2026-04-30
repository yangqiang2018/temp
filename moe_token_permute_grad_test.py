import torch_npu

try:
    from .moe_token_permute_grad import MoeTokenPermuteGrad as _OrigMoeTokenPermuteGrad
except ImportError:
    from moe_token_permute_grad import MoeTokenPermuteGrad as _OrigMoeTokenPermuteGrad


class MoeTokenPermuteGrad(_OrigMoeTokenPermuteGrad):
    def __call__(self, permuted_output_grad, sorted_indices):
        torch_npu.npu.synchronize()
        return super().__call__(permuted_output_grad, sorted_indices)
