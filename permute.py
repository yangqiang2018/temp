import logging
import os
import torch
import torch.nn as nn
import torch_npu

from ..registry import BaseKernel, KernelType, replace_methods

logger = logging.getLogger(__name__)

_DTYPE_MAP = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}

USE_TILELANG = os.getenv("EN_TILELANG", "0") == "1"


def _unwrap_async(t):
    if t is None:
        return t
    try:
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        if isinstance(t, AsyncCollectiveTensor):
            t = t.wait()
    except ImportError:
        pass
    return t


def _lazy_import_tilelang():
    from .moe_token_permute import MoeTokenPermute
    from .moe_token_permute_grad import MoeTokenPermuteGrad
    from .moe_token_unpermute import MoeTokenUnpermute
    from .moe_token_unpermute_grad import MoeTokenUnpermuteGrad

    return MoeTokenPermute, MoeTokenPermuteGrad, MoeTokenUnpermute, MoeTokenUnpermuteGrad


_op_cache = {}


def _get_op(cls, cache_key, **kwargs):
    if cache_key not in _op_cache:
        _op_cache[cache_key] = cls(**kwargs)
    return _op_cache[cache_key]


class _TileLangPermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, top_k, num_experts, layer_id):
        MoeTokenPermute, _, _, _ = _lazy_import_tilelang()

        x_local = _unwrap_async(x)
        indices_local = _unwrap_async(indices)

        num_tokens, hidden_size = x_local.shape
        dtype_str = _DTYPE_MAP[x_local.dtype]

        key = ("permute", layer_id, num_tokens, top_k, hidden_size, num_experts, dtype_str)
        op = _get_op(
            MoeTokenPermute, key, num_tokens=num_tokens, topK=top_k, hidden_size=hidden_size, num_experts=num_experts, dtype=dtype_str
        )

        indices_flat = indices_local.view(-1)
        permuted, sorted_indices = op(x_local, indices_flat)

        ctx.save_for_backward(sorted_indices)
        ctx.num_tokens = num_tokens
        ctx.top_k = top_k
        ctx.hidden_size = hidden_size
        ctx.num_experts = num_experts
        ctx.dtype_str = dtype_str
        ctx.layer_id = layer_id
        return permuted, sorted_indices

    @staticmethod
    def backward(ctx, grad_permuted, _grad_sorted):
        _, MoeTokenPermuteGrad, _, _ = _lazy_import_tilelang()
        (sorted_indices,) = ctx.saved_tensors

        sorted_indices = _unwrap_async(sorted_indices)
        grad_permuted = _unwrap_async(grad_permuted)

        key = ("permute_grad", ctx.layer_id, ctx.num_tokens, ctx.top_k, ctx.hidden_size, ctx.num_experts, ctx.dtype_str)
        op = _get_op(
            MoeTokenPermuteGrad,
            key,
            num_tokens=ctx.num_tokens,
            topK=ctx.top_k,
            hidden_size=ctx.hidden_size,
            num_experts=ctx.num_experts,
            dtype=ctx.dtype_str,
            TILE_H=ctx.hidden_size,
        )

        grad_input = op(grad_permuted, sorted_indices)

        return grad_input, None, None, None, None


class _TileLangUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed_output, sorted_indices, top_scores, num_tokens, top_k, layer_id):
        _, _, MoeTokenUnpermute, _ = _lazy_import_tilelang()

        ro_local = _unwrap_async(routed_output)
        si_local = _unwrap_async(sorted_indices)
        ts_local = _unwrap_async(top_scores) if top_scores is not None else None

        hidden_size = ro_local.shape[1]
        has_probs = ts_local is not None
        dtype_str = _DTYPE_MAP[ro_local.dtype]

        key = ("unpermute", layer_id, num_tokens, top_k, hidden_size, has_probs, dtype_str)
        op = _get_op(
            MoeTokenUnpermute, key, num_tokens=num_tokens, topK=top_k, hidden_size=hidden_size, has_probs=has_probs, dtype=dtype_str
        )

        result = op(ro_local, si_local, ts_local) if has_probs else op(ro_local, si_local)

        ctx.save_for_backward(routed_output, sorted_indices, top_scores)
        ctx.num_tokens = num_tokens
        ctx.top_k = top_k
        ctx.hidden_size = hidden_size
        ctx.has_probs = has_probs
        ctx.dtype_str = dtype_str
        ctx.layer_id = layer_id
        return result

    @staticmethod
    def backward(ctx, grad_output):
        _, _, _, MoeTokenUnpermuteGrad = _lazy_import_tilelang()
        routed_output, sorted_indices, top_scores = ctx.saved_tensors

        routed_output = _unwrap_async(routed_output)
        sorted_indices = _unwrap_async(sorted_indices)
        top_scores = _unwrap_async(top_scores) if top_scores is not None else None
        grad_output = _unwrap_async(grad_output)

        key = ("unpermute_grad", ctx.layer_id, ctx.num_tokens, ctx.top_k, ctx.hidden_size, ctx.has_probs, ctx.dtype_str)
        op = _get_op(
            MoeTokenUnpermuteGrad,
            key,
            num_tokens=ctx.num_tokens,
            topK=ctx.top_k,
            hidden_size=ctx.hidden_size,
            has_probs=ctx.has_probs,
            NUM_CORES=24,
            dtype=ctx.dtype_str,
        )

        if ctx.has_probs:
            grad_routed, grad_probs = op(routed_output, grad_output, sorted_indices, probs=top_scores)
            return grad_routed, None, grad_probs, None, None, None
        else:
            grad_routed = op(routed_output, grad_output, sorted_indices)
            return grad_routed, None, None, None, None, None


def _npu_moe_forward(self, x):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, self.expert_bias)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    if self.shared_experts is not None:
        out = self.shared_experts(x)
    else:
        out = torch.zeros_like(x)

    unpermuted = torch_npu.npu_moe_token_unpermute(routed_output, sorted_indices, top_scores.to(x.dtype))
    return (out + unpermuted).reshape(bs, slen, dim)


def _tilelang_moe_forward(self, x):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)
    num_tokens = x.shape[0]

    top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, self.expert_bias)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    num_experts = num_tokens_per_expert.shape[0]

    routed_input, sorted_indices = _TileLangPermute.apply(x, indices, self.reorderer.top_k, num_experts, id(self))

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    if self.shared_experts is not None:
        out = self.shared_experts(x)
    else:
        out = torch.zeros_like(x)

    sorted_indices_in = (
        sorted_indices.reshape(-1).contiguous().to(torch.int32) if not isinstance(sorted_indices, type(None)) else sorted_indices
    )
    top_scores_in = top_scores.to(x.dtype).contiguous()
    routed_output_in = routed_output.contiguous()

    unpermuted = _TileLangUnpermute.apply(routed_output_in, sorted_indices_in, top_scores_in, num_tokens, self.reorderer.top_k, id(self))

    return (out + unpermuted).reshape(bs, slen, dim)


class PermuteKernel(BaseKernel):
    kernel_type = KernelType.PERMUTE
    MOE_PACKAGE = "torchtitan.models.moe"

    @classmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:
        forward_fn = _tilelang_moe_forward if USE_TILELANG else _npu_moe_forward
        backend = "TileLang" if USE_TILELANG else "torch_npu"

        count = replace_methods("MoE", "forward", forward_fn, package=cls.MOE_PACKAGE)

        logger.info(f"  [Permute] Applied {count} replacement(s) using {backend} backend")

        return model
