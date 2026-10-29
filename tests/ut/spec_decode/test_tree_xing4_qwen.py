from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase
from vllm_ascend.attention.tree_spec import MlaTreeSpecAdapter, tree_spec_enabled
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.v2.spec_decode import (
    _is_tree_dspark,
    dsv4_dspark_draft,
    init_speculator,
)


class TestTreeXing4QwenCombination(TestBase):
    @staticmethod
    def _config():
        target = SimpleNamespace(
            architectures=["Xing4_0ForCausalLM"],
            hf_config=SimpleNamespace(
                architectures=["Xing4_0ForCausalLM"],
                model_type="deepseek_v3",
            ),
        )
        draft = SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            hf_config=SimpleNamespace(
                architectures=["Qwen3DSparkModel"],
                model_type="qwen3",
            ),
        )
        speculative = SimpleNamespace(
            use_dspark=lambda: True,
            num_speculative_tokens=7,
            draft_model_config=draft,
            target_model_config=target,
        )
        return SimpleNamespace(
            additional_config={},
            speculative_config=speculative,
            model_config=target,
        )

    def test_qwen_dspark_draft_is_not_misclassified_as_dsv4(self):
        vllm_config = self._config()
        speculative = vllm_config.speculative_config

        self.assertTrue(_is_tree_dspark(speculative))
        self.assertFalse(dsv4_dspark_draft(vllm_config, speculative))

    def test_init_speculator_selects_tree_host_for_xing4_qwen(self):
        vllm_config = self._config()
        device = object()
        with (
            patch(
                "vllm_ascend.worker.v2.spec_decode.dflash_tree_spec_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.v2.spec_decode.tree.speculator."
                "AscendTreeSpeculator"
            ) as tree_speculator,
        ):
            result = init_speculator(vllm_config, device)

        tree_speculator.assert_called_once_with(vllm_config, device)
        self.assertIs(result, tree_speculator.return_value)

    def test_xing4_mla_builder_configures_tree_mask_geometry(self):
        vllm_config = self._config()
        vllm_config.scheduler_config = SimpleNamespace(max_num_seqs=8)
        vllm_config.model_config.max_model_len = 4096
        builder = SimpleNamespace(attn_mask_builder=MagicMock())

        with (
            patch(
                "vllm_ascend.attention.tree_spec.tree_decode_threshold",
                return_value=33,
            ),
            patch(
                "vllm_ascend.ascend_config.get_ascend_config",
                return_value=SimpleNamespace(
                    vllm_config=vllm_config,
                    tree_spec_config=SimpleNamespace(
                        enabled=True,
                        method="beam",
                        topk=4,
                        budget=32,
                    ),
                ),
            ),
        ):
            self.assertTrue(tree_spec_enabled(vllm_config))
            MlaTreeSpecAdapter().configure_builder(builder, vllm_config, object())

        builder.attn_mask_builder.configure_tree_mask.assert_called_once_with(
            max_num_decode=8,
            query_len=33,
            kv_len=4096,
        )

    def test_xing4_non_dsa_mla_selects_mla_backend(self):
        selector = SimpleNamespace(
            use_mla=True,
            use_sparse=False,
            use_compress=False,
            use_batch_invariant=False,
        )
        with patch("vllm_ascend.platform.is_310p", return_value=False):
            backend = NPUPlatform.get_attn_backend_cls(None, selector)

        self.assertEqual(
            backend,
            "vllm_ascend.attention.mla_v1.AscendMLABackend",
        )
