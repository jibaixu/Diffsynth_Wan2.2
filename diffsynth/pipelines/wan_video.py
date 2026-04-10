import os
import torch, types
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from typing import Optional, Union
from typing_extensions import Literal

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_action_encoder import WanVideoActionEncoder
from ..models.wan_video_track_encoder import WanVideoActionTrackFuser, WanVideoTrackEncoder
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from .wan_video_spec import WanModuleSpec


class WanVideoPipeline(BasePipeline):

    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
        modules: Optional[list[str]] = None,
    ):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.module_spec = WanModuleSpec.parse(modules)
        self.modules = self.module_spec.modules
        self.text_mode = self.module_spec.text_mode
        self.enable_text = self.module_spec.enable_text
        self.enable_text_encoder = self.module_spec.enable_text_encoder
        self.clip_mode = self.module_spec.clip_mode
        self.action_injection_mode = self.module_spec.action_mode
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.action_encoder: WanVideoActionEncoder = None
        self.track_encoder: WanVideoTrackEncoder = None
        self.action_track_fuser: WanVideoActionTrackFuser = None
        self.vae: WanVideoVAE = None
        self.num_track_views = 1
        self.track_num_points_per_view = 256
        self.track_noise_std = 0.0
        self.track_clip_min = -0.25
        self.track_clip_max = 1.25
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]
        if self.enable_text:
            self.units.append(WanVideoUnit_PromptEmbedder())
        if self.action_injection_mode != "off":
            self.units.append(WanVideoUnit_ActionEmbedder())


    def model_fn(self, *args, **kwargs):
        return model_fn_wan_video(*args, action_injection_mode=self.action_injection_mode, **kwargs)


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        vram_limit: float = None,
        modules: Optional[list[str]] = None,
        track_num_points_per_view: int = 256,
        num_track_views: int = 1,
        track_noise_std: float = 0.0,
    ):
        module_spec = WanModuleSpec.parse(modules)
        modules = list(module_spec.modules)
        action_enabled = module_spec.action_enabled

        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors")
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(
            device=device,
            torch_dtype=torch_dtype,
            modules=modules,
        )
        pipe.track_num_points_per_view = max(1, int(track_num_points_per_view))
        pipe.num_track_views = max(1, int(num_track_views))
        pipe.track_noise_std = max(0.0, float(track_noise_std))
        model_kwargs_overrides = {
            "wan_video_dit": {
                "has_text_input": module_spec.has_text_input_for_dit,
                "use_text_embedding": module_spec.use_text_embedding,
            }
        }
        model_pool = pipe.download_and_load_models(
            model_configs,
            vram_limit,
            model_kwargs_overrides=model_kwargs_overrides,
        )

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder") if module_spec.enable_text_encoder else None
        pipe.dit = model_pool.fetch_model("wan_video_dit")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.action_encoder = model_pool.fetch_model("wan_video_action_encoder") if action_enabled else None
        pipe.track_encoder = model_pool.fetch_model("wan_video_track_encoder") if action_enabled else None
        pipe.action_track_fuser = model_pool.fetch_model("wan_video_action_track_fuser") if action_enabled else None

        if action_enabled:
            action_dim = 14
            dim = getattr(pipe.dit, "dim", 1536) if pipe.dit is not None else 1536
            track_dim = pipe.track_num_points_per_view * pipe.num_track_views * 2
            if pipe.action_encoder is None:
                if module_spec.action_mode == "adaln":
                    pipe.action_encoder = WanVideoActionEncoder(
                        action_dim=action_dim,
                        dim=dim,
                        num_action_per_chunk=81,
                        in_features=None,
                        hidden_features=dim * 4,
                    )
                else:
                    pipe.action_encoder = WanVideoActionEncoder(action_dim=action_dim, dim=dim)
            if pipe.track_encoder is None:
                if module_spec.action_mode == "adaln":
                    pipe.track_encoder = WanVideoTrackEncoder(
                        track_dim=track_dim,
                        dim=dim,
                        num_track_per_chunk=81,
                        in_features=None,
                        hidden_features=dim * 4,
                    )
                else:
                    pipe.track_encoder = WanVideoTrackEncoder(track_dim=track_dim, dim=dim)
            if pipe.action_track_fuser is None:
                pipe.action_track_fuser = WanVideoActionTrackFuser(dim=dim)

            pipe.action_encoder = pipe.action_encoder.to(dtype=pipe.torch_dtype, device=pipe.device)
            pipe.track_encoder = pipe.track_encoder.to(dtype=pipe.torch_dtype, device=pipe.device)
            pipe.action_track_fuser = pipe.action_track_fuser.to(dtype=pipe.torch_dtype, device=pipe.device)

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None and module_spec.enable_text_encoder:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        prompt_emb: Optional[Union[torch.Tensor, str, os.PathLike]] = None,
        negative_prompt_emb: Optional[Union[torch.Tensor, str, os.PathLike]] = None,
        # Unified video input (i2v uses T=1)
        input_video: Optional[torch.Tensor] = None,
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        num_history_frames=1,
        # Action conditioning
        action: Optional[torch.Tensor] = None,
        track: Optional[torch.Tensor] = None,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        if not isinstance(input_video, torch.Tensor) or input_video.ndim != 5:
            raise TypeError("`input_video` must be a torch.Tensor with shape (V, C, T, H, W).")

        input_video = input_video.to(dtype=self.torch_dtype, device=self.device)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "prompt_emb": prompt_emb,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "prompt_emb": negative_prompt_emb,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_video": input_video, "denoising_strength": denoising_strength,
            "num_views": int(input_video.shape[0]),
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames, "num_history_frames": num_history_frames,
            "action": action,
            "track": track,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Historical-frame conditioning path: keep history latents fixed during denoising.
        if int(input_video.shape[2]) > 1 and int(input_video.shape[2]) < int(num_frames):
            self.load_models_to_device(["vae"])
            history_latents_views = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            history_latents_views = history_latents_views.to(dtype=self.torch_dtype, device=self.device)
            history_latents = rearrange(history_latents_views, "v c t h w -> 1 c t (v h) w")
            history_t = ((num_history_frames - 1) // 4) + 1
            inputs_shared["history_latents"] = history_latents[:, :, :history_t]

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "history_latents" in inputs_shared:
                history_t = ((inputs_shared["num_history_frames"] - 1) // 4) + 1
                inputs_shared["latents"][:, :, :history_t] = inputs_shared["history_latents"][:, :, :history_t]
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # Decode
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"]
        num_views = int(inputs_shared.get("num_views", 1))
        if latents.shape[-2] % num_views != 0:
            raise ValueError(f"Latent height {latents.shape[-2]} is not divisible by num_views={num_views}.")

        latents_by_view = rearrange(latents, "b c t (v h) w -> (b v) c t h w", v=num_views, h=latents.shape[-2] // num_views)
        video = self.vae.decode(latents_by_view, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "num_history_frames"),
            output_params=("height", "width", "num_frames", "num_history_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, num_history_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames, "num_history_frames": num_history_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit): #infer & train
    """
    作用: 在扩散模型的潜在空间(latent space)中生成初始随机噪声
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, height, width, num_frames, seed, rand_device):
        num_views = int(input_video.shape[0])
        # 计算 VAE 潜在空间的帧数
        length = (num_frames - 1) // 4 + 1
        latent_height = (height * num_views) // pipe.vae.upsampling_factor
        latent_width = width // pipe.vae.upsampling_factor
        shape = (1, pipe.vae.model.z_dim, length, latent_height, latent_width)

        # noise: (B=1, C_vae=16, F_lat=F/4, H/8, W/8)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit): # no infer & train
    """
    作用: 将输入视频(像素空间)通过 VAE 编码到潜在空间
    用途: 用于 Video-to-Video 任务,在训练和推理阶段都会用到
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, num_frames, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        if int(input_video.shape[2]) <= 1:
            # i2v path: single-frame conditioning is handled by CLIP/VAE image embedders.
            return {"latents": noise}
        if (not pipe.scheduler.training) and int(input_video.shape[2]) < int(num_frames):
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = input_video.to(dtype=pipe.torch_dtype, device=pipe.device)

        # Encode each view independently in VAE batch, then concatenate views in latent height.
        input_latents_views = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        input_latents_views = input_latents_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = rearrange(input_latents_views, "v c t h w -> 1 c t (v h) w")

        if pipe.scheduler.training:
            # 训练模式: 返回纯噪声和编码后的 latents (用于计算损失)
            return {"latents": noise, "input_latents": input_latents}
        else:
            # 推理模式: 将噪声添加到编码后的 latents 上,形成扩散过程的起点
            # timesteps[0] 是扩散过程的第一个时间步 (噪声最多的时刻)
            # 这个操作相当于: noisy_latents = input_latents + noise_level * noise
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit): #infer & train
    """
    作用: 将文本提示词(prompt)编码为文本嵌入向量,作为扩散模型的条件
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True, 
            input_params_posi={"prompt": "prompt", "prompt_emb": "prompt_emb", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "prompt_emb": "prompt_emb", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        """
        1. 使用 tokenizer 将文本转换为 token IDs
        2. 使用 text_encoder 将 token IDs 编码为高维嵌入向量
        3. 清除 padding 位置的嵌入,避免影响注意力计算
        """
        if pipe.tokenizer is None or pipe.text_encoder is None:
            raise ValueError("Text encoder or tokenizer is not available. Please provide pre-extracted prompt embeddings or load the text encoder.")
        if prompt is None:
            raise ValueError("Prompt is None and no pre-extracted embedding is provided.")

        # 使用 tokenizer 将文本转换为 token IDs 和 attention mask
        # ids: (B=1, L_word=512) - token IDs,L 是序列长度 (最大512)
        # mask: (B=1, L_word=512) - attention mask,1表示有效token,0表示padding
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)

        # 计算每个样本的有效序列长度 (非padding的token数量)
        # seq_lens: (B,) - 每个样本的实际长度
        seq_lens = mask.gt(0).sum(dim=1).long()

        # prompt_emb: (B=1, L_word=512, D_text=4096)
        prompt_emb = pipe.text_encoder(ids, mask)

        # 将 padding 位置的嵌入向量置零
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt=None, positive=None, prompt_emb=None) -> dict:
        if prompt_emb is None:
            pipe.load_models_to_device(self.onload_model_names)
            prompt_emb = self.encode_prompt(pipe, prompt)
        else:
            if isinstance(prompt_emb, (str, os.PathLike)):
                prompt_emb = torch.load(prompt_emb, map_location="cpu", weights_only=False)
            if not isinstance(prompt_emb, torch.Tensor):
                prompt_emb = torch.as_tensor(prompt_emb)
            prompt_emb = prompt_emb.detach()
            prompt_emb = prompt_emb.to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"context": prompt_emb}



class WanVideoUnit_ActionEmbedder(PipelineUnit): # infer & train
    """
    作用: 将动作序列编码为条件嵌入,对齐到 VAE 的时间下采样尺度
    """
    def __init__(self):
        super().__init__(
            input_params=("action", "track", "num_frames", "input_video"),
            output_params=("action_emb",),
            onload_model_names=("action_encoder", "track_encoder", "action_track_fuser")
        )

    @staticmethod
    def _fit_sequence_length(sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = int(sequence.shape[1])
        if current_length == target_length:
            return sequence
        if current_length > target_length:
            return sequence[:, :target_length]
        if current_length <= 0:
            raise ValueError("Cannot pad an empty condition sequence.")
        pad_frames = target_length - current_length
        pad = sequence[:, -1:, ...].repeat(1, pad_frames, *([1] * (sequence.ndim - 2)))
        return torch.cat([sequence, pad], dim=1)

    @staticmethod
    def _chunk_condition_for_latents(sequence: torch.Tensor, num_frames: int) -> torch.Tensor:
        length = (num_frames - 1) // 4 + 1
        sequence = torch.concat(
            [torch.repeat_interleave(sequence[:, 0:1], repeats=4, dim=1), sequence[:, 1:]],
            dim=1,
        )
        return sequence.contiguous().view(sequence.shape[0], length, 4, *sequence.shape[2:])

    @staticmethod
    def _downsample_action_condition(sequence: torch.Tensor, num_frames: int) -> torch.Tensor:
        return WanVideoUnit_ActionEmbedder._chunk_condition_for_latents(sequence, num_frames).mean(dim=2)

    @staticmethod
    def _downsample_track_condition(sequence: torch.Tensor, num_frames: int) -> torch.Tensor:
        # Preserve real per-frame coordinates instead of averaging track positions across time.
        return WanVideoUnit_ActionEmbedder._chunk_condition_for_latents(sequence, num_frames).select(dim=2, index=3).contiguous()

    @staticmethod
    def _ensure_track_tensor(track: torch.Tensor, expected_points: int) -> torch.Tensor:
        if track.ndim == 4:
            if track.shape[-1] != 2:
                raise ValueError(f"`track` last dim must be 2, got {tuple(track.shape)}")
            track_points = int(track.shape[2])
            if track_points != expected_points:
                raise ValueError(f"`track` point count mismatch: expected {expected_points}, got {track_points}")
            return track
        if track.ndim == 3:
            expected_width = expected_points * 2
            if track.shape[-1] != expected_width:
                raise ValueError(f"`track` width mismatch: expected {expected_width}, got {int(track.shape[-1])}")
            return rearrange(track, "b t (n c) -> b t n c", c=2).contiguous()
        raise ValueError(f"`track` must have shape (B,T,N,2) or (B,T,N*2), got {tuple(track.shape)}")

    @staticmethod
    def _apply_track_noise(pipe: WanVideoPipeline, track: torch.Tensor) -> torch.Tensor:
        std = float(getattr(pipe, "track_noise_std", 0.0))
        if std <= 0.0 or not getattr(pipe.scheduler, "training", False):
            return track
        base_min = track.amin()
        base_max = track.amax()
        noise = torch.randn_like(track) * std
        clip_min = torch.minimum(track.new_tensor(pipe.track_clip_min), base_min - 0.05)
        clip_max = torch.maximum(track.new_tensor(pipe.track_clip_max), base_max + 0.05)
        return torch.clamp(track + noise, min=clip_min, max=clip_max)

    def process(self, pipe: WanVideoPipeline, action, track, num_frames, input_video=None) -> dict:
        if action is None:
            return {}
        if pipe.action_encoder is None:
            raise ValueError("Action encoder is not available in the pipeline.")
        if pipe.track_encoder is None:
            raise ValueError("Track encoder is not available in the pipeline.")
        if pipe.action_track_fuser is None:
            raise ValueError("Action-track fuser is not available in the pipeline.")
        if track is None:
            raise ValueError("`track` is required when the WAN action branch is enabled.")

        if input_video is not None and int(input_video.shape[0]) != int(pipe.num_track_views):
            raise ValueError(
                f"Input video view count {int(input_video.shape[0])} does not match initialized "
                f"track view count {int(pipe.num_track_views)}."
            )

        pipe.load_models_to_device(self.onload_model_names)
        action = torch.as_tensor(action, device=pipe.device, dtype=pipe.torch_dtype)
        expected_points = int(pipe.track_num_points_per_view) * int(pipe.num_track_views)
        track = torch.as_tensor(track, device=pipe.device, dtype=pipe.torch_dtype)
        track = self._ensure_track_tensor(track, expected_points)

        action = self._fit_sequence_length(action, int(num_frames))
        track = self._fit_sequence_length(track, int(num_frames))
        track = self._apply_track_noise(pipe, track)
        track = rearrange(track, "b t n c -> b t (n c)").contiguous()

        if pipe.action_injection_mode == "noise":
            action = self._downsample_action_condition(action, int(num_frames))
            track = self._downsample_track_condition(track, int(num_frames))

        if pipe.action_injection_mode == "adaln":
            action_frame_count = pipe.action_encoder.action_embedding[0].in_features // pipe.action_encoder.action_dim
            track_frame_count = pipe.track_encoder.action_embedding[0].in_features // pipe.track_encoder.track_dim
            if action_frame_count != track_frame_count:
                raise ValueError(
                    f"Action/track encoder frame mismatch: action={action_frame_count}, track={track_frame_count}"
                )
            action = self._fit_sequence_length(action, action_frame_count)
            track = self._fit_sequence_length(track, track_frame_count)
            action = rearrange(action, "b f d -> b (f d)").contiguous()
            track = rearrange(track, "b f d -> b (f d)").contiguous()

        action_emb = pipe.action_encoder(action)
        track_emb = pipe.track_encoder(track)
        action_emb = pipe.action_track_fuser(torch.cat([action_emb, track_emb], dim=-1))
        if pipe.action_injection_mode == "cross":
            action_pos = torch.arange(action_emb.shape[1], device=action_emb.device, dtype=action_emb.dtype)
            action_emb = action_emb + sinusoidal_embedding_1d(action_emb.shape[-1], action_pos).unsqueeze(0)
            
        return {"action_emb": action_emb}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit): #infer & train
    """
    CLIP 图像编码器单元
    作用: 使用 CLIP 模型将输入图像编码为高层语义特征,作为条件信息
    用途: Image-to-Video 任务
    特点: 提取的是图像的语义特征
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "height", "width", "num_history_frames"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, height, width, num_history_frames):
        if input_video is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding or not pipe.dit.has_image_input:
            return {}

        pipe.load_models_to_device(self.onload_model_names)
        current_frame = input_video[:, :, num_history_frames - 1]
        if pipe.clip_mode == 0:
            image = rearrange(current_frame, "v c h w -> 1 c (v h) w")
            # clip_context: (B=1, N_token=1[cls]+256=257, D_clip=1280)
            clip_context = pipe.image_encoder.encode_image([image])
        else:
            # Encode each view independently, then flatten tokens as a single prefix.
            clip_context = pipe.image_encoder.encode_image([current_frame])
            clip_context = rearrange(clip_context, "v n d -> 1 (v n) d")

        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit): #infer &train
    """
    VAE 图像编码器单元
    作用: 将输入的开始帧通过 VAE 编码为潜在表示,并附加掩码信息
    用途: Image-to-Video 任务
    特点: 提供像素级的条件信息,并通过掩码明确标记哪些帧是已知的条件帧
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "num_frames", "num_history_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, num_frames, num_history_frames, height, width, tiled, tile_size, tile_stride):
        if input_video is None or not pipe.dit.require_vae_embedding or not pipe.dit.has_image_input:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        view_images = input_video[:, :, :num_history_frames]
        num_views = int(view_images.shape[0])

        # 创建掩码 (mask),标记哪些帧是已知的条件帧
        # msk: (B=1, F=num_frames, H/8, W/8)
        msk = torch.zeros(1, num_frames, height//8, width//8, device=pipe.device, dtype=pipe.torch_dtype)
        msk[:, :num_history_frames] = 1

        # 调整掩码的形状以匹配 VAE 的时间下采样
        # VAE 在时间维度上有 4x 下采样,需要将掩码对齐到 latent 的时间维度
        # 步骤:
        # 1. 将第一帧重复4次 (因为VAE的temporal patch size是4)
        # msk: (B=1, F=num_frames + 3[第一帧重复3次], H/8, W/8)
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        # 3. 重新reshape: (1, F=num_frames + 3, H/8, W/8) -> (1, F/4, 4[VAE时间缩放], H/8, W/8)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        # 4. 转置并去掉batch维: (可见通道=4, F/4, H/8, W/8)
        msk = msk.transpose(1, 2)[0]
        # 扩展到多视角并在高度维拼接: (4, F/4, V*H/8, W/8)
        msk = msk.unsqueeze(0).repeat(num_views, 1, 1, 1, 1)
        msk = rearrange(msk, "v c t h w -> c t (v h) w")

        padding = torch.zeros(
            num_views, view_images.shape[1], int(num_frames) - int(num_history_frames), int(height), int(width),
            dtype=view_images.dtype,
            device=view_images.device,
        )
        vae_inputs = torch.cat([view_images, padding], dim=2)

        # 使用 VAE 编码器将输入编码到潜在空间
        # y_views: (V, C_vae=16, F/4, H/8, W/8)
        y_views = pipe.vae.encode(vae_inputs, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        y_views = y_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        # (V, C, T, H, W) -> (C, T, V*H, W)
        y = rearrange(y_views, "v c t h w -> c t (v h) w")

        # 将掩码和 VAE 编码后的特征拼接在通道维度
        # 拼接后: y: (C=4+16=20, F/4, H/8, W/8)
        # - 前4个通道: 掩码信息 (标记哪些帧是已知的)
        # - 后16个通道: VAE 编码的潜在表示
        # 这样模型可以明确知道哪些区域是条件输入,哪些需要生成
        y = torch.concat([msk, y])
        # 添加 batch 维度: (1, C=20, F/4, H/8, W/8)
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit): # infer & train
    """
    Encode the conditioning frame directly into latents for Wan2.2 TI2V.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "latents", "num_history_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, latents, num_history_frames, tiled, tile_size, tile_stride):
        if input_video is None or not getattr(pipe.dit, "fuse_vae_embedding_in_latents", False):
            return {}
        if int(input_video.shape[2]) < int(num_history_frames):
            raise ValueError(f"`num_history_frames` ({num_history_frames}) exceeds input video frames ({input_video.shape[2]}).")

        pipe.load_models_to_device(self.onload_model_names)
        frame = input_video[:, :, 0:1]
        frame = frame.to(dtype=pipe.torch_dtype, device=pipe.device)
        z_views = pipe.vae.encode(frame, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        z_views = z_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        z = rearrange(z_views, "v c t h w -> 1 c t (v h) w")
        latents[:, :, 0:1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action_emb: Optional[torch.Tensor] = None,
    action_injection_mode: str = "off",
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    fuse_vae_embedding_in_latents: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """
    - latents: (B=1, C=16, F/4, H/8, W/8) - 潜在空间的噪声
    - timestep: 当前扩散时间步 (0~1000)
    - context: (B=1, L_word=512, D_text=4096) - 文本s嵌入
    - action_emb: (B=1, F, D_model) or (B=1, F/4, D_model) or (B=1, D_model) - action 嵌入 (可选)
    - clip_feature: (B=1, N_token=257, D_clip=1280) - CLIP 图像特征 (可选)
    - y: (B=1, C=4[mask]+16[vae]=20, F/4, H/8, W/8) - mask+VAE 编码的首帧 (可选)
    - use_gradient_checkpointing: 是否使用梯度检查点 (节省显存)
    - use_gradient_checkpointing_offload: 是否将中间激活值 offload 到 CPU (进一步节省显存)
    """

    # ========== 步骤1: 时间步编码 ==========
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
    else:
        # t: (B=1, D_model=1536)
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))

    if action_injection_mode == "adaln" and action_emb is not None:
        if t.ndim == 3 and action_emb.ndim == 2:
            t = t + action_emb.unsqueeze(1)
        else:
            t = t + action_emb

    # 将时间嵌入投影为调制参数
    # t_mod: (B=1, 6, D_model=1536)
    # 这6个参数用于 AdaLN (Adaptive Layer Normalization) 调制
    if t.ndim == 3:
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    text_token_count = 0

    # ========== 步骤2: 文本条件处理 ==========
    # context: (B=1, L_word=512, D_text=4096) -> (B=1, L_word=512, D_model=1536)
    if getattr(dit, "use_text_embedding", getattr(dit, "has_text_input", True)) and context is not None:
        context = dit.text_embedding(context)
        text_token_count = context.shape[1]
    elif not getattr(dit, "has_text_input", True):
        context = None
    elif not getattr(dit, "use_text_embedding", True):
        context = None

    if action_injection_mode == "cross" and action_emb is not None:
        if context is None:
            context = action_emb
        else:
            context = torch.cat([context, action_emb], dim=1)
        text_token_count = context.shape[1]

    x = latents

    # ========== 步骤3: 图像条件整合 ==========
    # 3.1 整合 VAE 图像条件 (如果提供)
    if y is not None and dit.has_image_input and dit.require_vae_embedding:
        # x:加噪嵌入 与mask 首帧vae嵌入在通道维度拼接
        # x 加噪嵌入: (B=1, C=16, F/4, H/8, W/8) + y: VAE编码: (B=1, C_y=20, F/4, H/8, W/8) -> (B=1, C+C_y=36, F/4, H/8, W/8)
        # 这样模型可以同时看到噪声 latent 和条件图像的潜在表示
        x = torch.cat([x, y], dim=1)

    # 3.2 整合 CLIP 图像特征 (如果提供)
    if clip_feature is not None and dit.has_image_input and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        # context: (B=1, L_token=512, D_model=1536) + clip_emb: (B=1, N_img=257, D_model=1538) -> (B, L+N_img=769, D_model)
        if context is None:
            context = clip_embdding
        else:
            context = torch.cat([clip_embdding, context], dim=1)


    # ========== 步骤4: Patchify - 将3D体积转换为token序列 ==========
    # 将连续的潜在表示切分为不重叠的3D patch
    # x: (B=1, C+C_y=20, F/4, H/8, W/8) -> (B=1, D_model=1536, F/8, H/16, W/16)
    # - F_p=4, H_p=2, W_p=2: 单个patch所占 (时间、高度、宽度方向)
    x = dit.patchify(x)
    f, h, w = x.shape[2:]  # 记录 patch 的网格尺寸
    
    if action_injection_mode == "noise" and action_emb is not None:
        # action_emb: (B, F, D_model) -> (B, D_model, F, 1, 1), broadcast to (H/16, W/16)
        action_emb = rearrange(action_emb, "b f d -> b d f 1 1")
        action_emb = repeat(action_emb, "b d f 1 1 -> b d f h w", h=h, w=w)
        x = x + action_emb

    # 将3D patch grid 展平为1D token 序列
    # x: (B, D_model, F/4, H/16, W/16) -> (B, N总token数, D_model=1536)
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    # ========== 步骤5: 位置编码 (RoPE - Rotary Position Embedding) ==========
    # 为每个 token 生成3D位置编码 (时间、高度、宽度)
    # freqs: (N总token, 1, D_freq=64)
    # 每个 token 的位置编码由其在 (f, h, w) grid 中的坐标决定
    # RoPE 会在 attention 计算时旋转 query 和 key,从而注入位置信息
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),  # 时间维度的位置编码
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),  # 高度维度的位置编码
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)   # 宽度维度的位置编码
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # ========== 步骤6: Transformer Blocks - 去噪主循环 ==========
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block in dit.blocks:
        block.cross_attn.text_token_count = text_token_count
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            # 标准梯度检查点: 不保存中间激活值,反向传播时重新计算
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            # 正常前向传播 (最快,但显存占用最大)
            x = block(x, context, t_mod, freqs)

    # ========== 步骤7: 输出投影和 Unpatchify ==========
    # 7.1 使用最终的 head 层进行输出投影
    # 这里会再次使用时间步 t 进行调制,并投影到输出通道数
    # x: (B, N总token, D_model) -> (B, N总token, C_out=64)
    x = dit.head(x, t)

    # 7.2 将 token 序列重构回 3D 体积
    # x: (B, N, C_out=64) -> (B, C_vae=16, F/4, H/8, W/8)
    # 这就是模型预测的噪声
    x = dit.unpatchify(x, (f, h, w))

    return x
