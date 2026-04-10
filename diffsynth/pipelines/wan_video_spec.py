import os
import glob
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Union


WanTextMode = Literal["off", "t5", "emb"]
WanActionMode = Literal["off", "noise", "adaln", "cross"]
WanImageMode = Literal["off", "default", "flat"]

WAN_DEFAULT_MODULES = ("dit", "text", "vae", "image", "action")
WAN_MODULE_FILES = {
    "dit": ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.pth"),
    "text": ("models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-bf16.safetensors"),
    "vae": ("Wan2.1_VAE.pth", "Wan2.1_VAE.safetensors", "Wan2.2_VAE.pth", "Wan2.2_VAE.safetensors"),
    "image": (
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
    ),
}
WAN_TOKENIZER_SUBDIR = os.path.join("google", "umt5-xxl")
WanModelPath = Union[str, list[str]]


def wan_module_base(name: str) -> str:
    return str(name).partition(":")[0].strip().lower()


def _split_module_spec(module: str) -> tuple[str, Optional[str]]:
    module = str(module).strip().lower()
    if not module:
        raise ValueError("Empty WAN module spec is not allowed.")
    base, sep, variant = module.partition(":")
    if not base:
        raise ValueError(f"Invalid WAN module spec: {module!r}")
    return base.strip(), variant.strip() if sep else None


def _pick_wan_candidate(model_root: str, candidates: tuple[str, ...]) -> WanModelPath:
    for name in candidates:
        path = os.path.join(model_root, name)
        if os.path.isfile(path):
            return path
    # Diffusers sharded checkpoint support:
    # if index exists, load all shard files as a list.
    if "diffusion_pytorch_model.safetensors" in candidates:
        index_path = os.path.join(model_root, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.isfile(index_path):
            shard_paths = sorted(glob.glob(os.path.join(model_root, "diffusion_pytorch_model-*.safetensors")))
            if len(shard_paths) > 0:
                return shard_paths
    return os.path.join(model_root, candidates[0])


@dataclass(frozen=True)
class WanRuntimeConfig:
    modules: tuple[str, ...]
    module_bases: tuple[str, ...]
    text_mode: WanTextMode
    action_mode: WanActionMode
    image_mode: WanImageMode
    enable_text: bool
    enable_text_encoder: bool
    action_enabled: bool
    has_text_input_for_dit: bool
    clip_mode: int
    data_file_keys: tuple[str, ...]
    model_paths: tuple[WanModelPath, ...]
    tokenizer_path: Optional[str]


@dataclass(frozen=True)
class WanModuleSpec:
    modules: tuple[str, ...]
    text_mode: WanTextMode
    action_mode: WanActionMode
    image_mode: WanImageMode

    @classmethod
    def parse(cls, load_modules) -> "WanModuleSpec":
        if isinstance(load_modules, cls):
            return load_modules

        if not load_modules:
            raw_modules = list(WAN_DEFAULT_MODULES)
        elif isinstance(load_modules, str):
            raw_modules = [item.strip() for item in load_modules.split(",") if item.strip()]
        else:
            raw_modules = [str(item).strip() for item in load_modules if str(item).strip()]

        order: list[str] = []
        normalized_by_base: dict[str, str] = {}
        text_mode: WanTextMode = "off"
        action_mode: WanActionMode = "off"
        image_mode: WanImageMode = "default"

        def remember(base: str, normalized: str) -> None:
            if base not in normalized_by_base:
                order.append(base)
            normalized_by_base[base] = normalized

        def drop(base: str) -> None:
            if base in normalized_by_base:
                del normalized_by_base[base]
                order[:] = [item for item in order if item != base]

        for raw_module in raw_modules:
            base, variant = _split_module_spec(raw_module)

            if base == "text":
                if variant is None or variant == "t5":
                    remember("text", "text")
                    text_mode = "t5"
                    continue
                if variant == "emb":
                    remember("text", "text:emb")
                    text_mode = "emb"
                    continue
                if variant == "off":
                    drop("text")
                    text_mode = "off"
                    continue
                raise ValueError(f"Unsupported WAN text module spec: {raw_module!r}")

            if base == "action":
                if variant is None or variant == "noise":
                    remember("action", "action")
                    action_mode = "noise"
                    continue
                if variant in ("adaln", "cross"):
                    remember("action", f"action:{variant}")
                    action_mode = variant
                    continue
                if variant == "off":
                    drop("action")
                    action_mode = "off"
                    continue
                raise ValueError(f"Unsupported WAN action module spec: {raw_module!r}")

            if base == "image":
                if variant is None:
                    remember("image", "image")
                    image_mode = "default"
                    continue
                if variant == "flat":
                    remember("image", "image:flat")
                    image_mode = "flat"
                    continue
                if variant == "off":
                    drop("image")
                    image_mode = "off"
                    continue
                raise ValueError(f"Unsupported WAN image module spec: {raw_module!r}")

            if base in ("dit", "vae"):
                if variant is not None:
                    raise ValueError(f"Unsupported WAN module variant: {raw_module!r}")
                remember(base, base)
                continue

            raise ValueError(f"Unsupported WAN module spec: {raw_module!r}")

        return cls(
            modules=tuple(normalized_by_base[base] for base in order if base in normalized_by_base),
            text_mode=text_mode,
            action_mode=action_mode,
            image_mode=image_mode,
        )

    @property
    def module_bases(self) -> tuple[str, ...]:
        return tuple(wan_module_base(module) for module in self.modules)

    @property
    def enable_text(self) -> bool:
        return self.text_mode != "off"

    @property
    def enable_text_encoder(self) -> bool:
        return self.text_mode == "t5"

    @property
    def action_enabled(self) -> bool:
        return self.action_mode != "off"

    @property
    def has_text_input_for_dit(self) -> bool:
        return self.enable_text or self.action_mode == "cross"

    @property
    def use_text_embedding(self) -> bool:
        return self.enable_text

    @property
    def clip_mode(self) -> int:
        return 1 if self.image_mode == "flat" else 0

    @property
    def weight_modules(self) -> tuple[str, ...]:
        modules: list[str] = []
        for base in self.module_bases:
            if base == "text" and not self.enable_text_encoder:
                continue
            modules.append(base)
        return tuple(modules)

    def build_runtime(self, model_root: str, data_file_keys: Iterable[str]) -> WanRuntimeConfig:
        if not model_root:
            raise ValueError("`--model_paths` is required.")

        keys = [str(key).strip() for key in data_file_keys if str(key).strip()]

        def ensure(name: str) -> None:
            if name not in keys:
                keys.append(name)

        if not self.enable_text:
            keys = [key for key in keys if key not in ("prompt_emb", "negative_prompt_emb")]
        elif self.text_mode == "emb":
            ensure("prompt_emb")

        if self.action_enabled:
            ensure("action")
            ensure("track")
        else:
            keys = [key for key in keys if key not in ("action", "track")]

        model_paths: list[WanModelPath] = []
        for module in self.weight_modules:
            candidates = WAN_MODULE_FILES.get(module)
            if candidates is None:
                continue
            model_paths.append(_pick_wan_candidate(model_root, candidates))

        tokenizer_path = os.path.join(model_root, WAN_TOKENIZER_SUBDIR) if self.enable_text_encoder else None

        return WanRuntimeConfig(
            modules=self.modules,
            module_bases=self.module_bases,
            text_mode=self.text_mode,
            action_mode=self.action_mode,
            image_mode=self.image_mode,
            enable_text=self.enable_text,
            enable_text_encoder=self.enable_text_encoder,
            action_enabled=self.action_enabled,
            has_text_input_for_dit=self.has_text_input_for_dit,
            clip_mode=self.clip_mode,
            data_file_keys=tuple(keys),
            model_paths=tuple(model_paths),
            tokenizer_path=tokenizer_path,
        )
