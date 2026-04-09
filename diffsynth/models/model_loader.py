from ..core.loader import load_model, hash_model_file
from ..core.vram import AutoWrappedModule
from ..configs import MODEL_CONFIGS, VRAM_MANAGEMENT_MODULE_MAPS
import importlib, inspect, json, torch


class ModelPool:
    def __init__(self):
        self.model = []
        self.model_name = []
        self.model_path = []
        
    def import_model_class(self, model_class):
        split = model_class.rfind(".")
        model_resource, model_class = model_class[:split], model_class[split+1:]
        model_class = importlib.import_module(model_resource).__getattribute__(model_class)
        return model_class
    
    def need_to_enable_vram_management(self, vram_config):
        return vram_config["offload_dtype"] is not None and vram_config["offload_device"] is not None
    
    def fetch_module_map(self, model_class, vram_config):
        if self.need_to_enable_vram_management(vram_config):
            if model_class in VRAM_MANAGEMENT_MODULE_MAPS:
                module_map = {self.import_model_class(source): self.import_model_class(target) for source, target in VRAM_MANAGEMENT_MODULE_MAPS[model_class].items()}
            else:
                module_map = {self.import_model_class(model_class): AutoWrappedModule}
        else:
            module_map = None
        return module_map
    
    def load_model_file(self, config, path, vram_config, vram_limit=None, model_kwargs_override=None):
        model_class = self.import_model_class(config["model_class"])
        model_config = dict(config.get("extra_kwargs", {}))
        if model_kwargs_override:
            valid_args = set(inspect.signature(model_class.__init__).parameters.keys())
            valid_args.discard("self")
            for key, value in model_kwargs_override.items():
                if key in valid_args:
                    model_config[key] = value
        if "state_dict_converter" in config:
            state_dict_converter = self.import_model_class(config["state_dict_converter"])
        else:
            state_dict_converter = None
        module_map = self.fetch_module_map(config["model_class"], vram_config)
        model = load_model(
            model_class, path, model_config,
            vram_config["computation_dtype"], vram_config["computation_device"],
            state_dict_converter,
            use_disk_map=True,
            vram_config=vram_config, module_map=module_map, vram_limit=vram_limit,
        )
        return model
    
    def default_vram_config(self):
        vram_config = {
            "offload_dtype": None,
            "offload_device": None,
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cpu",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cpu",
        }
        return vram_config
    
    def auto_load_model(self, path, vram_config=None, vram_limit=None, clear_parameters=False, model_kwargs_overrides=None):
        if vram_config is None:
            vram_config = self.default_vram_config()
        model_hash = hash_model_file(path)
        loaded = False
        model_kwargs_overrides = model_kwargs_overrides or {}
        for config in MODEL_CONFIGS:
            if config["model_hash"] == model_hash:
                override = model_kwargs_overrides.get(config["model_name"])
                model = self.load_model_file(config, path, vram_config, vram_limit=vram_limit, model_kwargs_override=override)
                if clear_parameters: self.clear_parameters(model)
                self.model.append(model)
                model_name = config["model_name"]
                self.model_name.append(model_name)
                self.model_path.append(path)
                model_info = {"model_name": model_name, "model_class": config["model_class"], "extra_kwargs": config.get("extra_kwargs")}
                loaded = True
        if not loaded:
            raise ValueError(f"Cannot detect the model type. File: {path}. Model hash: {model_hash}")
    
    def fetch_model(self, model_name, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            model = None
        elif len(fetched_models) == 1:
            model = fetched_models[0]
        else:
            if index is None:
                model = fetched_models[0]
            elif isinstance(index, int):
                model = fetched_models[:index]
            else:
                model = fetched_models
        return model

    def clear_parameters(self, model: torch.nn.Module):
        for name, module in model.named_children():
            self.clear_parameters(module)
        for name, param in model.named_parameters(recurse=False):
            setattr(model, name, None)
