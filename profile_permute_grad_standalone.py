import torch
import torch_npu
import warnings

from moe_token_permute_grad import MoeTokenPermuteGrad

warnings.filterwarnings("ignore", message="Cannot create tensor with interal format")


def main(num_tokens=8 * 1024, hidden_size=7168, topk=8, num_experts=256):
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

    for _ in range(10):
        tl_grad_op(grad_permuted_tokens, sorted_indices)
    torch_npu.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_standalone"),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(15):
            tl_grad_op(grad_permuted_tokens, sorted_indices)
            prof.step()
        torch_npu.npu.synchronize()

    print("Standalone profile written to ./prof_standalone/")
    print("Kernel Output Match")


if __name__ == "__main__":
    main()
