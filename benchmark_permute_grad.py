import torch
import torch_npu
from moe_token_permute_grad import MoeTokenPermuteGrad
import warnings

warnings.filterwarnings("ignore", message="Cannot create tensor with interal format")


def benchmark_moe_permute_grad(num_tokens, hidden_size, topk, num_experts, warmup=10, iters=100):
    """
    针对指定的规格对 MoeTokenPermuteGrad 和 torch_npu 原生反向算子进行 Benchmark。
    """
    tokens = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="npu", requires_grad=True)
    indices = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="npu")

    npu_permuted, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices)

    grad_permuted_tokens = torch.randn_like(npu_permuted)

    tl_grad_op = MoeTokenPermuteGrad(
        num_tokens=num_tokens,
        topK=topk,
        hidden_size=hidden_size,
        num_experts=num_experts,
        dtype="float16",
    )

    start_tl = torch_npu.npu.Event(enable_timing=True)
    end_tl = torch_npu.npu.Event(enable_timing=True)

    for _ in range(warmup):
        tl_grad_op(grad_permuted_tokens, sorted_indices)
    torch_npu.npu.synchronize()

    start_tl.record()
    for _ in range(iters):
        tl_grad_op(grad_permuted_tokens, sorted_indices)
    end_tl.record()
    torch_npu.npu.synchronize()

    tl_avg_time = start_tl.elapsed_time(end_tl) / iters

    start_torch = torch_npu.npu.Event(enable_timing=True)
    end_torch = torch_npu.npu.Event(enable_timing=True)

    def run_torch_backward():
        torch.autograd.grad(
            outputs=npu_permuted,
            inputs=tokens,
            grad_outputs=grad_permuted_tokens,
            retain_graph=True,
        )

    for _ in range(warmup):
        run_torch_backward()
    torch_npu.npu.synchronize()

    start_torch.record()
    for _ in range(iters):
        run_torch_backward()
    end_torch.record()
    torch_npu.npu.synchronize()

    torch_avg_time = start_torch.elapsed_time(end_torch) / iters

    torch_npu.npu.empty_cache()

    return tl_avg_time, torch_avg_time


if __name__ == "__main__":
    configs = [
        (8 * 1024, 7168, 8, 256, "super large"),
        (4 * 1024, 7168, 8, 256, "large"),
        (16, 16, 4, 4, "small"),
    ]

    print("Starting MoE Token Permute GRAD Scenario Benchmarks...")
    print("-" * 115)
    print(
        f"{'Scenario':<30} | {'Tokens':<6} | {'Hidden':<6} | {'TopK':<4} | {'Experts':<7} | {'TileLang (ms)':<13} | {'Torch (ms)':<10} | {'Speedup'}"
    )
    print("-" * 115)

    for config in configs:
        num_tokens, hidden_size, topk, num_experts, desc = config

        try:
            tl_time, torch_time = benchmark_moe_permute_grad(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                topk=topk,
                num_experts=num_experts,
                warmup=10,
                iters=100,
            )

            speedup = torch_time / tl_time if tl_time > 0 else 0.0

            print(
                f"{desc:<30} | {num_tokens:<6} | {hidden_size:<6} | {topk:<4} | {num_experts:<7} | "
                f"{tl_time:<13.3f} | {torch_time:<10.3f} | {speedup:.2f}x"
            )

        except Exception as e:
            print(f"{desc:<30} | {num_tokens:<6} | {hidden_size:<6} | {topk:<4} | {num_experts:<7} | {'FAILED':<13} | {'FAILED':<10} | N/A")
            print(f"  -> Error: {e}")
