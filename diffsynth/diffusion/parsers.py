import argparse

_DEFAULT_GROUP_TITLES = {"positional arguments", "optional arguments", "options"}

def _get_group(parser: argparse.ArgumentParser, title: str):
    for group in parser._action_groups:
        if group.title == title:
            return group
    return parser.add_argument_group(title)

def build_grouped_config(parser: argparse.ArgumentParser, args):
    if args is None:
        return None
    args_dict = args if isinstance(args, dict) else vars(args)
    grouped = {}
    used = set()
    for group in parser._action_groups:
        if group.title in _DEFAULT_GROUP_TITLES:
            continue
        values = {}
        for action in group._group_actions:
            dest = getattr(action, "dest", None)
            if dest in args_dict:
                values[dest] = args_dict[dest]
                used.add(dest)
        if values:
            grouped[group.title] = values
    other = {k: args_dict[k] for k in args_dict if k not in used}
    if other:
        grouped["other"] = other
    return grouped

def add_dataset_base_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "dataset")
    group.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    group.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    group.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    group.add_argument("--dataset_num_workers", type=int, default=8, help="Number of workers for data loading.")
    group.add_argument("--data_file_keys", type=str, default="video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "image")
    group.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--max_pixels", type=int, default=4096*4096, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "video")
    group.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    group.add_argument("--spatial_division_factor", type=int, choices=[16, 32], default=16, help="Spatial alignment factor for both height and width.")
    group.add_argument("--max_pixels", type=int, default=4096*4096, help="Maximum number of pixels per frame, used for dynamic resolution.")
    group.add_argument("--resize_mode", type=str, default="fit", choices=["crop", "fit"], help="Resize behavior: crop (center crop), fit (no crop), short (scale by short edge).")
    group.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    group.add_argument("--num_history_frames", type=int, default=1, help="Number of conditioning history frames. Must satisfy 1 <= num_history_frames < num_frames.")
    group.add_argument("--history_template_sampling", type=int, choices=[0, 1], default=0, help="Enable history-template sampling: frame0 + stride anchors(4x repeat) + contiguous future.")
    group.add_argument("--history_anchor_stride", type=int, default=8, help="Backward stride for history anchors when history_template_sampling=1.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "model")
    group.add_argument("--model_paths", type=str, default=None, help="Root path of the WAN pretrained weights.")
    group.add_argument("--load_modules", type=str, default=None, help="Comma-separated modules to load: dit,text,vae,image,action,track. Supported variants: action:noise|adaln|cross|off, track:residual|off, text:t5|emb|off, and image:flat|off. You can also set the default via env LOAD_MODULES.")
    group.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    group.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    group.add_argument("--extra_inputs", default="input_image", help="Additional model inputs, comma-separated.")
    group.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    group.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "training")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    group.add_argument("--seed", type=int, default=42, help="Random seed for python/numpy/torch.")
    group.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    group.add_argument("--trainable_models", type=str, default="dit", help="Models to train, e.g., dit, vae, text_encoder, track_context_adapter.")
    group.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    group.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    group.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    group.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    group.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "output")
    group.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    group.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    group.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    group.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint (.safetensors) used to initialize training weights (model-only resume).")
    group.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint directory saved by accelerator (e.g., output_path/epoch-0).")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "lora")
    group.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    group.add_argument("--lora_target_modules", type=str, default=None, help="Which layers LoRA is added to (default: q,k,v,o,ffn.0,ffn.2).")
    group.add_argument("--lora_rank", type=int, default=None, help="Rank of LoRA.")
    group.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    group.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    group.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "gradient")
    group.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    group.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    group.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm for clipping.")
    return parser

def add_tracking_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "tracking")
    group.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Enable Weights & Biases tracking (1 启用，0 关闭).")
    group.add_argument("--use_swanlab", type=int, choices=[0, 1], default=0, help="Enable SwanLab tracking (1 启用，0 关闭).")
    group.add_argument("--swanlab_experiment_name", type=str, default=None, help="SwanLab experiment name. Defaults to output_path.")
    return parser

def add_action_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "action")
    group.add_argument("--action_type", type=str, choices=["state_joint", "state_pose", "action_joint", "action_pose"], default=None, help="Which action/state slice to load from parquet.")
    group.add_argument("--action_stat_path", type=str, default=None, help="Path to action/state normalization stats (stat.json). Defaults to dataset_base_path/meta/stat.json if present.")
    group.add_argument("--track_num_points_per_view", type=int, default=256, help="Number of cotracker points to sample per view from the first-frame visible set.")
    group.add_argument("--track_num_views", type=int, default=None, help="Optional override for the number of track/video views. When omitted, training infers it from metadata.")
    group.add_argument("--track_noise_std", type=float, default=0.01, help="Std of Gaussian noise applied to sampled cotracker coordinates during training. Set to 0 to disable.")
    return parser

def add_infer_config(parser: argparse.ArgumentParser):
    group = _get_group(parser, "infer")
    group.add_argument("--ckpt_path", dest="checkpoint_path", type=str, default=None, help=("Path to checkpoint file or directory (optional; merged onto pretrained WAN weights). " "Supports checkpoints that include dit/action_encoder/track_context_adapter keys."))
    group.add_argument("--atm_ckpt_path", type=str, default=None, help="Optional ATM checkpoint used to generate multiview track conditioning during inference.")
    group.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale for generation")
    group.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    group.add_argument("--sigma_shift", type=float, default=5.0, help="Sigma shift for WAN scheduler inference.")
    group.add_argument("--negative_prompt", type=str, default=("The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion"), help="Negative prompt for generation")
    group.add_argument("--negative_prompt_emb", type=str, default="prompt_emb/neg_prompt.pt", help="Path to the pre-extracted negative prompt embedding.")
    group.add_argument("--quality", type=int, default=5, help="Output video quality.")
    group.add_argument("--metrics", dest="enable_metrics", type=int, default=1, choices=[0, 1], help="Enable (1) or disable (0) evaluation metrics")
    group.add_argument("--metric_preset", type=str, default="core", choices=["core", "all"], help="Metric preset: core runs psnr/ssim/mse/lpips/fid/fvd only; all also runs PBench.")
    group.add_argument("--streaming_eval", type=int, default=1, choices=[0, 1], help="Enable streaming metric evaluation to avoid caching all decoded comparison videos in RAM.")
    group.add_argument("--eval_video_batch_size", type=int, default=1, help="Number of comparison videos to decode per batch during metric evaluation.")
    group.add_argument("--eval_num_workers", type=int, default=8, help="Worker threads for PSNR/SSIM/MSE during metric evaluation.")
    group.add_argument("--chunk_infer", type=int, default=1, choices=[0, 1], help="Enable chunked inference with 81-frame segments (0=off, 1=on).")
    group.add_argument("--sample_indices", type=str, default=None, help="Optional comma-separated sample indices for debugging or partial evaluation.")
    group.add_argument("--fps", type=int, default=30, help="Output video FPS")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_tracking_config(parser)
    return parser
