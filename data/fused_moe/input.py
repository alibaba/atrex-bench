import torch


def _make_inputs(
    token_count: int, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int
) -> dict[str, torch.Tensor]:
    hidden_states = torch.randn(token_count, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.1
    w1 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    ) * hidden_size ** (-0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    ) * intermediate_size ** (-0.5)
    # Realistic top-k routing: select top_k experts per token from router logits so that
    # each token's selected experts are unique (real MoE never routes a token to the same
    # expert twice). Using torch.randint here instead would produce duplicate experts per
    # token, an unphysical case that sorted-scatter MoE kernels handle differently from the
    # gather reference, spuriously failing accuracy checks.
    router_logits = torch.randn(token_count, num_experts, dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }
