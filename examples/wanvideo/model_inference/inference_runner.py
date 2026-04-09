"""
机器人视频推理 CLI 入口
负责：命令行参数解析、配置管理、启动推理引擎
"""
import argparse
import logging
import os

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


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Robot inference with WanVideo pipeline")
    parser = add_model_config(parser)
    parser = add_dataset_base_config(parser)
    parser = add_action_config(parser)
    parser = add_video_size_config(parser)
    parser = add_training_config(parser)
    parser = add_infer_config(parser)

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
