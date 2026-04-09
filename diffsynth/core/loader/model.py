from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict
import inspect
import torch


class _StateDictView:
    def __init__(self, state_dict, key_filter=None):
        self.state_dict = state_dict
        self.key_filter = key_filter

    def __iter__(self):
        for key in self.state_dict:
            if self.key_filter is None or self.key_filter(key):
                yield key

    def __getitem__(self, key):
        return self.state_dict[key]

    def __contains__(self, key):
        if key not in self.state_dict:
            return False
        return self.key_filter is None or self.key_filter(key)

    def __len__(self):
        return sum(1 for _ in self.__iter__())

    def keys(self):
        return list(self.__iter__())

    def items(self):
        for key in self:
            yield key, self[key]

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


def _converter_supports_key_filter(state_dict_converter):
    try:
        return "key_filter" in inspect.signature(state_dict_converter).parameters
    except (TypeError, ValueError):
        return False


def _convert_state_dict(state_dict, state_dict_converter, target_key_filter=None):
    if state_dict_converter is not None:
        if _converter_supports_key_filter(state_dict_converter):
            state_dict = state_dict_converter(state_dict, key_filter=target_key_filter)
        else:
            state_dict = state_dict_converter(state_dict)
            if target_key_filter is not None:
                state_dict = {k: v for k, v in state_dict.items() if target_key_filter(k)}
    else:
        state_dict = {
            key: state_dict[key]
            for key in state_dict
            if target_key_filter is None or target_key_filter(key)
        }
    return state_dict


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None):
    config = {} if config is None else config
    # Why do we use `skip_model_initialization`?
    # It skips the random initialization of model parameters,
    # thereby speeding up model loading and avoiding excessive memory usage.
    with skip_model_initialization():
        model = model_class(**config)
    target_key_filter = getattr(model, "should_load_state_dict_key", None)
    target_key_filter = target_key_filter if callable(target_key_filter) else None
    source_key_filter = getattr(model, "should_load_source_state_dict_key", None)
    source_key_filter = source_key_filter if callable(source_key_filter) else None
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            state_dict = DiskMap(path, device, torch_dtype=dtype)
            state_dict = _StateDictView(state_dict, key_filter=source_key_filter)
            state_dict = _convert_state_dict(state_dict, state_dict_converter, target_key_filter=target_key_filter)
            model.load_state_dict(state_dict, assign=True)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device, key_filter=source_key_filter)
        state_dict = _StateDictView(state_dict, key_filter=source_key_filter)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        state_dict = _convert_state_dict(state_dict, state_dict_converter, target_key_filter=target_key_filter)
        model.load_state_dict(state_dict, assign=True)
        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        model = model.to(dtype=torch_dtype, device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model
