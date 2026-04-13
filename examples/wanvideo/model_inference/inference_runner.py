"""
机器人视频推理 CLI 入口
负责：命令行参数解析、配置管理、启动推理引擎
"""
import argparse
import logging
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def debug_on():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.argv = [
        "examples/wanvideo/model_inference/inference_runner.py",
        "--dataset_base_path", "/data_jbx/Datasets/Realbot/4_4_four_tasks_wan",
        "--dataset_metadata_path", "/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/episodes_val.jsonl",
        "--action_stat_path", "/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/stat.json",
        "--action_type", "action_pose",
        "--height", "480",
        "--width", "640",
        "--num_frames", "17",
        "--num_history_frames", "1",
        "--spatial_division_factor", "32",
        "--model_paths", "/data1/modelscope/models/Wan-AI/Wan2.2-TI2V-5B",
        "--load_modules", "dit,text:emb,vae,image:off,action:noise",
        "--ckpt_path", "Ckpt/wan_atm001B_480_640_202604101603/epoch-49/epoch-49.safetensors",
        "--atm_ckpt_path", "/data_jbx/Codes/ATM/results/track_transformer/0409_realbot_track_transformer_001B_action_bs_16_grad_acc_4_numtrack_256_ep1001_0047/model_best.ckpt",
        "--negative_prompt_emb", "prompt_emb/neg_prompt.pt",
        "--metric_preset", "core",      #! ["core", "all"] 评测指标预设，core 包含核心指标，all 包含全部指标(core + PBench)
        "--sample_indices", "319",    #! 指定要推理的样本索引，默认推理全部样本。可以是单个索引（如 "319"）或逗号分隔的索引列表（如 "0,5,10"）。
        "--fps", "15",
        "--chunk_infer", "1",
        "--resume",                     #! 断点重连
        "--output_dir", "/data_jbx/Codes/Diffsynth_Wan2.2/Ckpt/wan_atm001B_480_640_202604101603/epoch-49/epoch-49/04M12D_21H37Min", #! 断点重连
        # "--eval_video_batch_size", "80", #! 评估时视频批量大小，避免内存占用过高
        # "--eval_num_workers", "8",      #! 评估时数据加载线程数，避免内存占用过高
    ]
    # Latest residual-adapter config in train.py:
    # "--load_modules", "dit,text:emb,vae,image:off,action:noise,track:residual",
    # Replace --ckpt_path with a trained residual checkpoint when available.
debug_on()


from diffsynth.diffusion.parsers import (
    add_action_config,
    add_dataset_base_config,
    add_infer_config,
    add_model_config,
    add_training_config,
    add_video_size_config,
    build_grouped_config,
)

from inference_support import (
    build_wan_inference_config,
    destroy_distributed_inference,
    initialize_distributed_inference,
    load_flat_config_defaults,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Robot inference with WanVideo pipeline")
    parser = add_model_config(parser)
    parser = add_dataset_base_config(parser)
    parser = add_action_config(parser)
    parser = add_video_size_config(parser)
    parser = add_training_config(parser)
    parser = add_infer_config(parser)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Use together with --resume to reuse an interrupted inference run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --output_dir by reusing any video files that are already present.",
    )

    for action in parser._actions:
        if getattr(action, "dest", None) == "dataset_base_path":
            action.required = False
            break

    pre_args, _ = parser.parse_known_args()
    defaults = load_flat_config_defaults(pre_args.checkpoint_path)
    if defaults:
        known = {action.dest for action in parser._actions if getattr(action, "dest", None)}
        parser.set_defaults(**{key: value for key, value in defaults.items() if key in known})

    args = parser.parse_args()
    if not args.dataset_base_path:
        raise ValueError("`--dataset_base_path` is required (or provide a config.json next to the checkpoint).")
    if args.resume and not args.output_dir:
        raise ValueError("`--resume` requires `--output_dir` to point to an existing inference directory.")

    grouped_config = build_grouped_config(parser, args) or {}
    return build_wan_inference_config(
        vars(args).copy(),
        grouped_config=grouped_config,
    )


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def main() -> None:
    from inference_engine import InferenceEngine

    config = parse_args()
    logger = setup_logging()
    dist_context = None
    try:
        dist_context = initialize_distributed_inference(logger)
        InferenceEngine(config, logger, dist_context=dist_context).run()
    finally:
        destroy_distributed_inference(dist_context)


if __name__ == "__main__":
    main()
