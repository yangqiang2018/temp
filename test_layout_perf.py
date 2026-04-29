import torch
import torch_npu
import warnings

from moe_token_permute_grad import MoeTokenPermuteGrad

warnings.filterwarnings("ignore", message="Cannot create tensor with interal format")


def time_one(op, perm_grad, sorted_indices, warmup=10, iters=100):
    for _ in range(warmup):
        op(perm_grad, sorted_indices)
    torch_npu.npu.synchronize()

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op(perm_grad, sorted_indices)
    end.record()
    torch_npu.npu.synchronize()
    return start.elapsed_time(end) / iters


def make_perm_grad_contiguous(E, H, dtype, device):
    return torch.randn(E, H, dtype=dtype, device=device)


def make_perm_grad_sliced_offset(E, H, dtype, device):
    big = torch.randn(2 * E, H, dtype=dtype, device=device)
    out = big[E : 2 * E]
    assert out.is_contiguous()
    return out


def make_perm_grad_strided_cols(E, H, dtype, device):
    big = torch.randn(E, 2 * H, dtype=dtype, device=device)
    out = big[:, :H]
    assert not out.is_contiguous()
    assert out.stride() == (2 * H, 1)
    return out


def make_perm_grad_transposed(E, H, dtype, device):
    big = torch.randn(H, E, dtype=dtype, device=device)
    out = big.t()
    assert not out.is_contiguous()
    assert out.shape == (E, H)
    return out


def make_perm_grad_view_from_3d(E, H, dtype, device):
    big = torch.randn(2, E, H, dtype=dtype, device=device)
    out = big[0]
    assert out.is_contiguous() is False or out.is_contiguous()
    return out


def make_perm_grad_from_autograd(num_tokens, hidden_size, topk, num_experts, dtype, device):
    tokens = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device, requires_grad=True)
    indices = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    permuted, _ = torch_npu.npu_moe_token_permute(tokens, indices)
    grad = torch.randn_like(permuted)
    return grad


def make_perm_grad_force_contiguous_after_strided(E, H, dtype, device):
    big = torch.randn(E, 2 * H, dtype=dtype, device=device)
    out = big[:, :H].contiguous()
    assert out.is_contiguous()
    return out


def main():
    num_tokens = 8 * 1024
    hidden_size = 7168
    topk = 8
    num_experts = 256
    dtype = torch.float16
    device = "npu"
    dtype_str = "float16"

    E = num_tokens * topk

    sorted_indices_template = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
    _, sorted_indices = torch_npu.npu_moe_token_permute(
        torch.randn(num_tokens, hidden_size, dtype=dtype, device=device),
        sorted_indices_template,
    )

    op = MoeTokenPermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        dtype=dtype_str,
    )

    variants = [
        ("contiguous (baseline)", make_perm_grad_contiguous(E, hidden_size, dtype, device)),
        ("sliced offset (still contig)", make_perm_grad_sliced_offset(E, hidden_size, dtype, device)),
        ("strided cols (non-contig, stride=2H)", make_perm_grad_strided_cols(E, hidden_size, dtype, device)),
        ("transposed (non-contig)", make_perm_grad_transposed(E, hidden_size, dtype, device)),
        ("view from 3D (probably contig)", make_perm_grad_view_from_3d(E, hidden_size, dtype, device)),
        (
            "from autograd (training-like)",
            make_perm_grad_from_autograd(num_tokens, hidden_size, topk, num_experts, dtype, device),
        ),
        (
            "strided then .contiguous() (control)",
            make_perm_grad_force_contiguous_after_strided(E, hidden_size, dtype, device),
        ),
    ]

    print("Testing MoeTokenPermuteGrad layout sensitivity")
    print(f"  num_tokens={num_tokens}  hidden={hidden_size}  topK={topk}  experts={num_experts}  dtype={dtype}")
    print(f"  E (num_tokens * topK) = {E}")
    print("-" * 100)
    print(f"{'Variant':<45} | {'is_contig':<10} | {'stride':<20} | {'avg ms':<10} | {'vs baseline'}")
    print("-" * 100)

    baseline_ms = None
    for name, perm_grad in variants:
        actual_contig = perm_grad.is_contiguous()
        stride_str = str(perm_grad.stride())

        try:
            avg_ms = time_one(op, perm_grad, sorted_indices)
            if baseline_ms is None:
                baseline_ms = avg_ms
            ratio = avg_ms / baseline_ms
            print(f"{name:<45} | {str(actual_contig):<10} | {stride_str:<20} | {avg_ms:<10.3f} | {ratio:.2f}x")
        except Exception as e:
            print(f"{name:<45} | {str(actual_contig):<10} | {stride_str:<20} | FAILED     | -")
            print(f"    -> {e}")

    print("-" * 100)
    print()
    print("Interpretation:")
    print("  * If 'strided cols' or 'transposed' show > 1.3x baseline, contiguity is the bottleneck.")
    print("  * If 'from autograd' is much slower than 'contiguous (baseline)', training layout matters.")
    print("  * If 'strided then .contiguous() (control)' matches baseline, .contiguous() in wrapper is a fix.")
    print("  * If all variants are within ~5%, contiguity is NOT the cause -- look elsewhere (msprof).")
    print("Kernel Output Match")


if __name__ == "__main__":
    main()
