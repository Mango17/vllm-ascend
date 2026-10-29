from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.attention.tree_spec import (
    tree_query_len,
    tree_spec_enabled,
)
from vllm_ascend.compilation.compiler_interface import (
    _compute_decode_cudagraph_batch_sizes,
)
from vllm_ascend.patch.platform.patch_spec_decode_stats import (
    _tree_verify_width,
)
from vllm_ascend.worker.v2.spec_decode import tree_target_query_len


class TestTreeVerifyWidth(TestBase):
    def _scheduler(self, *, enforce_eager: bool) -> SimpleNamespace:
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=enforce_eager)
            ),
            num_spec_tokens=7,
        )

    def test_tree_width_does_not_depend_on_eager(self):
        tree_config = SimpleNamespace(budget=16, topk=2)
        ascend_config = SimpleNamespace(tree_spec_config=tree_config)
        with (
            patch(
                "vllm_ascend.patch.platform.patch_spec_decode_stats."
                "dflash_tree_spec_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend.patch.platform.patch_spec_decode_stats."
                "get_ascend_config",
                return_value=ascend_config,
            ),
        ):
            self.assertEqual(
                _tree_verify_width(self._scheduler(enforce_eager=False)),
                16,
            )
            self.assertEqual(
                _tree_verify_width(self._scheduler(enforce_eager=True)),
                16,
            )

    def test_chain_width_is_unchanged(self):
        tree_config = SimpleNamespace(budget=16, topk=1)
        ascend_config = SimpleNamespace(tree_spec_config=tree_config)
        with (
            patch(
                "vllm_ascend.patch.platform.patch_spec_decode_stats."
                "dflash_tree_spec_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend.patch.platform.patch_spec_decode_stats."
                "get_ascend_config",
                return_value=ascend_config,
            ),
        ):
            scheduler = self._scheduler(enforce_eager=False)
            self.assertEqual(_tree_verify_width(scheduler), scheduler.num_spec_tokens)

    def test_tree_capture_sizes_include_single_request_bucket(self):
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=7),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=[7, 8, 14, 16, 21, 24, 28],
                max_cudagraph_capture_size=28,
            ),
        )
        with (
            patch(
                "vllm_ascend.worker.v2.spec_decode.dflash_tree_spec_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.v2.spec_decode.tree_target_query_len",
                return_value=17,
            ),
        ):
            self.assertIn(
                17,
                _compute_decode_cudagraph_batch_sizes(vllm_config),
            )

    def test_width_one_qwen_uses_plain_dspark(self):
        vllm_config = self._spec_config(
            draft_arches=["Qwen3DSparkModel"],
            num_spec=7,
        )
        with (
            patch(
                "vllm_ascend.ascend_config.get_ascend_config",
                return_value=SimpleNamespace(
                    vllm_config=vllm_config,
                    tree_spec_config=SimpleNamespace(
                        enabled=True,
                        topk=1,
                        budget=32,
                    ),
                ),
            ),
            patch(
                "vllm.config.get_current_vllm_config",
                return_value=vllm_config,
            ),
        ):
            self.assertFalse(tree_spec_enabled(vllm_config))
            self.assertIsNone(tree_query_len())
            self.assertIsNone(tree_target_query_len(vllm_config))

    def test_width_two_qwen_keeps_tree_path(self):
        vllm_config = self._spec_config(
            draft_arches=["Qwen3DSparkModel"],
            num_spec=7,
        )
        with patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=SimpleNamespace(
                vllm_config=vllm_config,
                tree_spec_config=SimpleNamespace(
                    enabled=True,
                    topk=2,
                    budget=16,
                )
            ),
        ):
            self.assertTrue(tree_spec_enabled(vllm_config))

    def test_width_one_dsv4_keeps_chain_tree(self):
        vllm_config = self._spec_config(
            draft_arches=["DSparkDraftModel"],
            num_spec=7,
        )
        with (
            patch(
                "vllm_ascend.ascend_config.get_ascend_config",
                return_value=SimpleNamespace(
                    vllm_config=vllm_config,
                    tree_spec_config=SimpleNamespace(
                        enabled=True,
                        topk=1,
                        budget=32,
                    )
                ),
            ),
            patch(
                "vllm.config.get_current_vllm_config",
                return_value=vllm_config,
            ),
        ):
            self.assertTrue(tree_spec_enabled(vllm_config))
            self.assertEqual(tree_query_len(), 8)
            self.assertEqual(tree_target_query_len(vllm_config), 8)

    def test_width_one_dflash_keeps_existing_tree_path(self):
        vllm_config = self._spec_config(
            draft_arches=["DFlashDraftModel"],
            num_spec=7,
            use_dspark=False,
        )
        with patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=SimpleNamespace(
                vllm_config=vllm_config,
                tree_spec_config=SimpleNamespace(
                    enabled=True,
                    topk=1,
                    budget=16,
                ),
            ),
        ):
            self.assertTrue(tree_spec_enabled(vllm_config))
            self.assertEqual(tree_query_len(), 17)
            self.assertEqual(tree_target_query_len(vllm_config), 17)

    @staticmethod
    def _spec_config(draft_arches, num_spec, *, use_dspark=True):
        draft = SimpleNamespace(
            architectures=draft_arches,
            hf_config=SimpleNamespace(
                architectures=draft_arches,
                model_type="",
            ),
        )
        target = SimpleNamespace(architectures=["Xing4_0ForCausalLM"], hf_config=None)
        return SimpleNamespace(
            additional_config={},
            speculative_config=SimpleNamespace(
                use_dspark=lambda: use_dspark,
                num_speculative_tokens=num_spec,
                draft_model_config=draft,
                target_model_config=target,
            ),
            model_config=target,
        )
