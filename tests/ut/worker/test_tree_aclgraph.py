from types import SimpleNamespace

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

import vllm_ascend.worker.v2.spec_decode as spec_decode
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


def test_reshape_tree_verify_descriptors_uses_uniform_tree_width(monkeypatch):
    monkeypatch.setattr(
        spec_decode,
        "dflash_tree_spec_enabled",
        lambda _config: True,
    )
    monkeypatch.setattr(
        spec_decode,
        "tree_target_query_len",
        lambda _config: 33,
    )

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=66,
        num_reqs=2,
        uniform_token_count=33,
    )
    manager = object.__new__(ModelAclGraphManager)
    manager.vllm_config = SimpleNamespace()
    manager._capture_descs = {CUDAGraphMode.FULL: [desc]}
    manager._candidates = {(66, 0): [desc]}

    manager._reshape_tree_verify_full_descs()

    reshaped = manager._capture_descs[CUDAGraphMode.FULL][0]
    assert reshaped.num_tokens == 66
    assert reshaped.num_reqs == 2
    assert reshaped.uniform_token_count == 33
    assert manager._candidates[(66, 0)] == [reshaped]
