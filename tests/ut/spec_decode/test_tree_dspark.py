import torch

from tests.ut.base import TestBase
from vllm_ascend.worker.v2.spec_decode.tree.beam import BeamTreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    greedy_tree_reject_torch,
)


def _layout(budget: int, num_reqs: int = 1) -> TreeLayout:
    return TreeLayout(
        tokens=torch.full((num_reqs, budget), -1, dtype=torch.long),
        depths=torch.zeros((num_reqs, budget), dtype=torch.int32),
        parents=torch.full((num_reqs, budget), -1, dtype=torch.int32),
        num_nodes=torch.zeros(num_reqs, dtype=torch.int32),
        visibility=torch.zeros((num_reqs, budget, budget), dtype=torch.bool),
        first_child=torch.full((num_reqs, budget + 1), -1, dtype=torch.int32),
        next_sibling=torch.full((num_reqs, budget + 1), -1, dtype=torch.int32),
    )


class TestBeamTreeBuilder(TestBase):
    def test_pool_keeps_every_parent_child_edge(self):
        builder = BeamTreeBuilder(budget=8, topk=3)
        self.assertEqual(builder._pool_capacity(3, 4), 30)

    def test_budget_selection_is_topological_and_prefix_closed(self):
        budget = 7
        builder = BeamTreeBuilder(budget=budget, topk=2)
        draft_logits = torch.tensor(
            [
                [
                    [4.0, 3.0, 2.0, 1.0],
                    [4.0, 3.0, 2.0, 1.0],
                    [4.0, 3.0, 2.0, 1.0],
                ]
            ]
        )
        tree = builder.build(
            draft_logits,
            _layout(budget),
            root_token_ids=torch.tensor([0]),
            graph_safe=True,
        )

        self.assertEqual(int(tree.num_nodes[0]), budget)
        parents = tree.parents[0].tolist()
        for node, parent in enumerate(parents, start=1):
            self.assertGreaterEqual(parent, 0)
            self.assertLess(parent, node)

    def test_stateful_markov_step_advances_selected_frontier(self):
        class StatefulMarkovModel:
            def __init__(self):
                self.state_inputs = []
                self.step_hidden = []

            def markov_init_state(self, batch_size):
                return torch.zeros((batch_size, 2), dtype=torch.float32)

            def markov_step(self, state, token_ids, hidden_states):
                self.state_inputs.append(state.clone())
                self.step_hidden.append(hidden_states.clone())
                bias = torch.zeros((token_ids.shape[0], 4), dtype=torch.float32)
                bias[:, token_ids % 4] = 20.0
                return state + 1.0, bias

        model = StatefulMarkovModel()
        builder = BeamTreeBuilder(budget=6, topk=2, draft_model=model)
        draft_logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        hidden = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
        tree = builder.build(
            draft_logits,
            _layout(6),
            root_token_ids=torch.tensor([1]),
            draft_hidden=hidden,
            graph_safe=True,
        )

        self.assertEqual(len(model.state_inputs), 3)
        self.assertEqual(len(model.step_hidden), 3)
        self.assertEqual(int(tree.num_nodes[0]), 6)
        for depth, hidden_states in enumerate(model.step_hidden):
            self.assertTrue(
                torch.equal(
                    hidden_states,
                    hidden[0, depth].expand(hidden_states.shape[0], -1),
                )
            )


class TestGreedyTreeRejection(TestBase):
    def test_follows_matching_children_then_emits_bonus(self):
        tree = _layout(3)
        tree.tokens[0] = torch.tensor([7, 8, 9])
        tree.first_child[0, 0] = 1
        tree.first_child[0, 1] = 2
        tree.first_child[0, 2] = 3

        target_token_ids = torch.tensor([7, 8, 9, 99])
        target_logits = torch.full((1, 4, 100), -100.0)
        target_logits[0, torch.arange(4), target_token_ids] = 100.0
        sampled = torch.full((1, 4), -1, dtype=torch.long)
        path = torch.full((1, 3), -1, dtype=torch.long)

        result = greedy_tree_reject_torch(
            tree,
            target_logits,
            num_speculative_tokens=3,
            path_node_ids=path,
            sampled_token_ids=sampled,
        )

        self.assertEqual(result[0].tolist(), [7, 8, 9, 99])
        self.assertEqual(path[0].tolist(), [1, 2, 3])

    def test_stops_at_first_missing_child(self):
        tree = _layout(2)
        tree.tokens[0] = torch.tensor([7, 8])
        tree.first_child[0, 0] = 1
        tree.next_sibling[0, 1] = 2

        target_token_ids = torch.tensor([7, 9, 6])
        target_logits = torch.full((1, 3, 100), -100.0)
        target_logits[0, torch.arange(3), target_token_ids] = 100.0
        sampled = torch.full((1, 3), -1, dtype=torch.long)
        path = torch.full((1, 2), -1, dtype=torch.long)

        result = greedy_tree_reject_torch(
            tree,
            target_logits,
            num_speculative_tokens=2,
            path_node_ids=path,
            sampled_token_ids=sampled,
        )

        self.assertEqual(result[0].tolist(), [7, 9, -1])
        self.assertEqual(path[0].tolist(), [1, -1])
