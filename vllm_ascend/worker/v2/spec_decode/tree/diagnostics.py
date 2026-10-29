from vllm.logger import init_logger

from vllm_ascend import envs

logger = init_logger("vllm.tree_spec")

TREE_DEBUG_SAMPLE_INTERVAL = 100
TREE_DEBUG_INITIAL_SAMPLES = 5


def tree_debug_enabled() -> bool:
    return envs.VLLM_ASCEND_TREE_DEBUG


def should_log_tree_debug(call_index: int) -> bool:
    return (
        call_index <= TREE_DEBUG_INITIAL_SAMPLES
        or call_index % TREE_DEBUG_SAMPLE_INTERVAL == 0
    )


def log_tree_debug(message: str, *args) -> None:
    if tree_debug_enabled():
        logger.info(message, *args)
