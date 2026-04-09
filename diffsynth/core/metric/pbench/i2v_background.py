import io
import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from dreamsim import dreamsim
from pbench.utils_i2v import load_video, load_i2v_dimension_info, dreamsim_transform, dreamsim_transform_Image
from pbench.distributed import distribute_list_to_rank, gather_list_of_dict, get_rank, print0
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def i2v_background(dream_model, video_pair_list, device):
    video_results = []
    sim_list = []

    max_weight = 0.4
    mean_weight = 0.3
    min_weight = 0.3

    image_transform = dreamsim_transform_Image(224)
    frames_transform = dreamsim_transform(224)

    for image_path, video_path in tqdm(video_pair_list):
        # input image preprocess & extract feature
        input_image = image_transform(Image.open(image_path))
        input_image = input_image.unsqueeze(0)
        input_image = input_image.to(device)
        input_image_features = dream_model.embed(input_image)
        input_image_features = F.normalize(input_image_features, dim=-1, p=2)

        # get frames from video
        images = load_video(video_path)
        images = frames_transform(images)

        # calculate sim between input image and frames in generated video
        conformity_scores = []
        consec_scores = []
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                image_features = dream_model.embed(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i != 0:
                    sim_consec = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    consec_scores.append(sim_consec)
                sim_to_input = max(0.0, F.cosine_similarity(input_image_features, image_features).item())
                conformity_scores.append(sim_to_input)
                former_image_features = image_features

        video_score = max_weight * np.max(conformity_scores) + \
            mean_weight * np.mean(consec_scores) + \
            min_weight * np.min(consec_scores)

        sim_list.append(video_score)
        video_results.append({'image_path': image_path, 'video_path': video_path, 'video_results': video_score})
    return np.mean(sim_list), video_results


def compute_i2v_background(json_dir, device, submodules_list, **kwargs):
    # 设置 dreamsim 缓存目录到项目目录，避免使用当前目录的 ./models
    from pbench.utils_i2v import CACHE_DIR
    dreamsim_cache_dir = os.path.join(CACHE_DIR, 'dreamsim_model')
    os.makedirs(dreamsim_cache_dir, exist_ok=True)

    # DreamSim 内部会把 torch.hub 目录切到 dreamsim_cache_dir，因此本地 DINO repo/权重
    # 需要放到这个目录下才能真正避开 GitHub 校验和重复下载。
    local_dino_repo = os.path.join(CACHE_DIR, 'dino_model', 'facebookresearch_dino_main')
    dreamsim_dino_repo = os.path.join(dreamsim_cache_dir, 'facebookresearch_dino_main')
    if os.path.exists(local_dino_repo) and not os.path.exists(dreamsim_dino_repo):
        try:
            os.symlink(local_dino_repo, dreamsim_dino_repo)
            print0(f"✓ 使用本地 DINO 仓库: {local_dino_repo} -> {dreamsim_dino_repo}")
        except OSError as e:
            print0(f"⚠ 无法创建 DINO 仓库符号链接: {e}，将继续使用默认加载逻辑")

    local_dino_ckpt = os.path.join(CACHE_DIR, 'dino_model', 'dino_vitbase16_pretrain.pth')
    dreamsim_ckpt_dir = os.path.join(dreamsim_cache_dir, 'checkpoints')
    os.makedirs(dreamsim_ckpt_dir, exist_ok=True)
    dreamsim_dino_ckpt = os.path.join(dreamsim_ckpt_dir, 'dino_vitbase16_pretrain.pth')
    if os.path.exists(local_dino_ckpt) and not os.path.exists(dreamsim_dino_ckpt):
        try:
            os.symlink(local_dino_ckpt, dreamsim_dino_ckpt)
            print0(f"✓ 使用本地 DINO 权重: {local_dino_ckpt} -> {dreamsim_dino_ckpt}")
        except OSError as e:
            print0(f"⚠ 无法创建 DINO 权重符号链接: {e}，将继续使用默认加载逻辑")

    dream_model, preprocess = dreamsim(pretrained=True, cache_dir=dreamsim_cache_dir)
    resolution = submodules_list['resolution']
    print0("Initialize DreamSim success")

    # Load all data
    video_pair_list, _ = load_i2v_dimension_info(json_dir, dimension='i2v_background', lang='en', resolution=resolution)
    print0(f"Total video pairs to process: {len(video_pair_list)}")

    # Distribute data across GPUs
    local_video_pair_list = distribute_list_to_rank(video_pair_list)
    print0(f"Rank {get_rank()} processing {len(local_video_pair_list)} video pairs")

    # Process local data
    local_score, local_video_results = i2v_background(dream_model, local_video_pair_list, device)

    # Gather results from all GPUs
    all_video_results = gather_list_of_dict(local_video_results)

    # Calculate overall score from all results
    if get_rank() == 0:
        all_scores = [result['video_results'] for result in all_video_results]
        overall_score = np.mean(all_scores)
        print0(f"Overall score calculated from {len(all_scores)} video pairs")
        return overall_score, all_video_results
    else:
        return local_score, local_video_results
