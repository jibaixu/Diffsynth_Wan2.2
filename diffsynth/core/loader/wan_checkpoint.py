import os
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .file import load_keys_dict, load_state_dict


ActionPrefix = "pipe.action_encoder."
UnsupportedPrefixes = ("action_encoder.", "dit.")


@dataclass(frozen=True)
class WanCheckpointStats:
    dit_key_count: int
    action_key_count: int
    ignored_key_count: int


def _classify_checkpoint_key(
    key: str,
    dit_key_filter: Optional[Callable[[str], bool]] = None,
) -> tuple[Literal["dit", "action_encoder", "ignore", "skip"], Optional[str]]:
    if key.startswith(ActionPrefix):
        return "action_encoder", key[len(ActionPrefix):]
    if key.startswith(UnsupportedPrefixes):
        return "ignore", None
    if key.startswith("pipe."):
        return "ignore", None
    if dit_key_filter is not None and not dit_key_filter(key):
        return "skip", None
    return "dit", key


def load_wan_checkpoint_into_pipeline(
    pipe,
    ckpt_path,
    torch_dtype=None,
    device: str = "cpu",
    logger=None,
    message_prefix: Optional[str] = None,
):
    ckpt_path = os.fspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if torch_dtype is None:
        torch_dtype = getattr(pipe, "torch_dtype", None)

    log_info = getattr(logger, "info", None) if logger is not None else None
    log_warning = getattr(logger, "warning", None) if logger is not None else None
    if not callable(log_info):
        log_info = print
    if not callable(log_warning):
        log_warning = log_info

    dit = getattr(pipe, "dit", None)
    dit_key_filter = getattr(dit, "should_load_state_dict_key", None)
    dit_key_filter = dit_key_filter if callable(dit_key_filter) else None

    if message_prefix is not None:
        log_info(f"{message_prefix}: {ckpt_path}")

    ignored_key_count = sum(
        1
        for key in load_keys_dict(ckpt_path)
        if _classify_checkpoint_key(key)[0] == "ignore"
    )

    def keep_checkpoint_key(key: str) -> bool:
        target, _ = _classify_checkpoint_key(key, dit_key_filter)
        return target in ("dit", "action_encoder")

    state_dict = load_state_dict(
        ckpt_path,
        torch_dtype=torch_dtype,
        device=device,
        key_filter=keep_checkpoint_key,
    )

    dit_state = {}
    action_state = {}
    for key, value in state_dict.items():
        target, normalized_key = _classify_checkpoint_key(key, dit_key_filter)
        if target == "dit":
            dit_state[normalized_key] = value
        elif target == "action_encoder":
            action_state[normalized_key] = value

    def load_component(name: str, module, component_state: dict) -> None:
        if not component_state:
            return
        if module is None:
            log_warning(f"  - {name} weights found ({len(component_state)} keys), but pipeline.{name} is None")
            return
        load_result = module.load_state_dict(component_state, strict=False)
        log_info(
            f"  - Loaded {name} keys: {len(component_state)} "
            f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
        )

    load_component("dit", getattr(pipe, "dit", None), dit_state)
    load_component("action_encoder", getattr(pipe, "action_encoder", None), action_state)

    if ignored_key_count > 0:
        log_info(f"  - Ignored {ignored_key_count} keys with unsupported checkpoint prefixes")

    return WanCheckpointStats(
        dit_key_count=len(dit_state),
        action_key_count=len(action_state),
        ignored_key_count=ignored_key_count,
    )
