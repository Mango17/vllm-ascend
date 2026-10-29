"""Debug-only reference implementation of the TreeSpark candidate policy."""

import torch


def _reference_markov_step(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
    depth_hidden: torch.Tensor,
    frontier_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply one parent-conditioned step without using the beam builder."""
    if draft_model is None:
        return depth_logits.unsqueeze(0).expand(frontier_tokens.numel(), -1), None

    markov_head = getattr(getattr(draft_model, "model", draft_model), "markov_head", None)
    head_type = getattr(markov_head, "head_type", "vanilla")
    if head_type == "vanilla":
        embeddings = draft_model.markov_embed(frontier_tokens)
        bias = draft_model.markov_bias(embeddings)
        return depth_logits.unsqueeze(0) + bias, None

    markov_step = draft_model.markov_step
    batch = frontier_tokens.numel()
    hidden = depth_hidden.unsqueeze(0).expand(batch, -1)
    states = (
        frontier_states
        if frontier_states is not None
        else draft_model.markov_init_state(batch)
    )
    new_states, bias = markov_step(states, frontier_tokens, hidden)
    return depth_logits.unsqueeze(0) + bias, new_states


def build_reference_tree(
    draft_model,
    *,
    draft_logits: torch.Tensor,
    draft_hidden: torch.Tensor,
    root_token_ids: torch.Tensor,
    top_k: int,
    max_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one request using the original TreeSpark per-slot loop."""
    assert draft_logits.shape[0] == 1
    device = draft_logits.device
    depth_count, vocab_size = draft_logits.shape[1:]
    width_limit = max(1, min(int(top_k), int(vocab_size)))

    pool_tokens = torch.empty(0, dtype=torch.long, device=device)
    pool_scores = torch.empty(0, dtype=torch.float32, device=device)
    pool_parents = torch.empty(0, dtype=torch.long, device=device)
    pool_depths = torch.empty(0, dtype=torch.long, device=device)

    frontier_tokens = root_token_ids[:1]
    frontier_scores = torch.zeros(1, dtype=torch.float32, device=device)
    frontier_pool = torch.full((1,), -1, dtype=torch.long, device=device)
    frontier_states = None
    if draft_model is not None:
        markov_init_state = getattr(draft_model, "markov_init_state", None)
        if markov_init_state is not None:
            frontier_states = markov_init_state(1)

    for depth in range(depth_count):
        step_logits, next_states = _reference_markov_step(
            draft_model,
            draft_logits[0, depth],
            frontier_tokens.unsqueeze(0),
            draft_hidden[0, depth],
            frontier_states.unsqueeze(0) if frontier_states is not None else None,
        )
        step_logits = step_logits[0]
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        top_vals, top_ids = log_probs.topk(width_limit, dim=-1)
        if draft_model is not None:
            top_ids = draft_model.map_draft_to_target(top_ids)
        candidate_scores = frontier_scores.unsqueeze(1) + top_vals
        num_candidates = frontier_tokens.size(0) * width_limit

        pool_parents = torch.cat(
            [pool_parents, frontier_pool.repeat_interleave(width_limit)]
        )
        pool_tokens = torch.cat([pool_tokens, top_ids.reshape(-1)])
        pool_scores = torch.cat([pool_scores, candidate_scores.reshape(-1)])
        pool_depths = torch.cat(
            [
                pool_depths,
                torch.full(
                    (num_candidates,),
                    depth,
                    dtype=torch.long,
                    device=device,
                ),
            ]
        )

        keep = min(width_limit, num_candidates)
        selected = candidate_scores.reshape(-1).topk(keep).indices
        frontier_tokens = top_ids.reshape(-1)[selected]
        frontier_scores = candidate_scores.reshape(-1)[selected]
        frontier_pool = (pool_tokens.size(0) - num_candidates) + selected
        if next_states is not None:
            frontier_states = next_states[0, selected // width_limit]

    num_pool = int(pool_tokens.size(0))
    num_selected = min(int(max_nodes), num_pool)
    order = torch.argsort(pool_depths, stable=True)
    order = order[torch.argsort(pool_scores[order], descending=True, stable=True)]
    packed = order[:num_selected].sort().values
    num_nodes = int(packed.size(0))

    remap = torch.zeros(num_pool, dtype=torch.long, device=device)
    remap[packed] = torch.arange(1, num_nodes + 1, device=device)
    pooled_parents = pool_parents[packed]
    non_root = pooled_parents >= 0
    parent_indices = torch.zeros(num_nodes, dtype=torch.long, device=device)
    parent_indices[non_root] = remap[pooled_parents[non_root]]

    return (
        pool_tokens[packed],
        parent_indices,
        pool_depths[packed] + 1,
    )
