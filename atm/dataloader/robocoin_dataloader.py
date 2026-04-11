import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import decord
from einops import rearrange
from tqdm import tqdm

from atm.dataloader.utils import ImgTrackColorJitter, ImgViewDiffTranslationAug
from atm.utils.flow_utils import sample_tracks_visible_first, sample_tracks_nearest_to_grids

class RoboCoinATMDataset(Dataset):
    def __init__(self,
                 jsonl_path,
                 dataset_dir,
                 img_size,
                 num_track_ts,
                 num_track_ids,
                 frame_stack=1,
                 cache_all=False,
                 cache_image=False,
                 cache_track=False,
                 num_demos=None,
                 vis=False,
                 aug_prob=0.,
                 augment_track=True,
                 views=None,
                 extra_state_keys=None,
                 uniform_sample=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.jsonl_path = jsonl_path
        
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.frame_stack = frame_stack
        
        # 缓存控制
        self.cache_all = cache_all
        self.cache_image = cache_image
        self.cache_track = cache_track
        
        self.num_demos = num_demos
        self.vis = vis
        self.aug_prob = aug_prob
        self.augment_track = augment_track
        self.uniform_sample = uniform_sample
        
        # 视角和额外状态处理
        self.views = views
        if self.views is not None:
            self.views.sort()
        self.extra_state_keys = extra_state_keys if extra_state_keys is not None else []

        if not self.cache_all:
            assert not self.cache_image, "cache_image is only supported when cache_all is True."
            assert not self.cache_track, "cache_track is only supported when cache_all is True."

        # 读取 JSONL 数据条目
        self.data_entries = self._load_jsonl(self.jsonl_path)
        if self.num_demos is not None:
            assert 0 < self.num_demos <= 1, "num_demos means the ratio of training data."
            n_demo = int(len(self.data_entries) * self.num_demos)
            self.data_entries = self.data_entries[:n_demo]
            
        print(f"Loaded {len(self.data_entries)} segments from {jsonl_path}")

        # 数据增强初始化
        self.augmentor = transforms.Compose([
            ImgTrackColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ImgViewDiffTranslationAug(input_shape=img_size, translation=8, augment_track=self.augment_track),
        ])

        # 缓存字典初始化
        self._cache = {}
        if self.cache_all:
            self._build_cache()

    def _load_jsonl(self, path):
        entries = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line.strip()))
        return entries

    def _build_cache(self):
        print("Building RAM cache for dataset...")
        for i, entry in enumerate(tqdm(self.data_entries)):
            cache_dict = {}
            action_path = os.path.join(self.dataset_dir, entry["action"])
            task_embed_path = os.path.join(self.dataset_dir, entry["task_embed"])

            # 默认缓存极小的数据：相对动作和文本特征
            cache_dict['actions'] = np.stack(pd.read_parquet(action_path)['action'].values)
            cache_dict['task_emb'] = torch.load(task_embed_path, map_location="cpu")

            # 按需缓存 Track
            if self.cache_track:
                track_path = os.path.join(self.dataset_dir, entry["track"])
                npz_data = np.load(track_path)
                cache_dict['tracks'] = npz_data['tracks']
                cache_dict['vis'] = npz_data['vis']

            # 按需缓存 Image
            if self.cache_image:
                video_path = os.path.join(self.dataset_dir, entry["video"])
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                frames = vr.get_batch(range(len(vr))).asnumpy()
                cache_dict['frames'] = frames

            self._cache[i] = cache_dict

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, index):
        entry = self.data_entries[index]
        start_frame = entry["start_frame"]
        
        # --- 1. 读取 Frame Stack 图像 ---
        if self.cache_all and self.cache_image:
            all_frames = self._cache[index]['frames']
        else:
            video_path = os.path.join(self.dataset_dir, entry["video"])
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            all_frames = vr

        img_start_idx = max(start_frame + 1 - self.frame_stack, 0)
        img_end_idx = start_frame + 1
        
        video_len = len(all_frames)
        img_indices = np.arange(img_start_idx, img_end_idx)
        img_indices = np.clip(img_indices, a_min=None, a_max=video_len - 1)

        if isinstance(all_frames, np.ndarray):
            frames = torch.from_numpy(all_frames[img_indices]).float()
        else:
            frames = torch.from_numpy(all_frames.get_batch(img_indices).asnumpy()).float()

        frames = rearrange(frames, "t h w c -> t c h w")

        if len(frames) < self.frame_stack:
            padding_frames = torch.zeros((self.frame_stack - len(frames), *frames.shape[1:]))
            frames = torch.cat([padding_frames, frames], dim=0)

        if frames.shape[2:] != self.img_size:
            frames = F.interpolate(frames, size=self.img_size, mode="bilinear", align_corners=False)

        # --- 2. 读取 Actions 和 Task Emb ---
        if self.cache_all:
            actions_all = self._cache[index]['actions']
            task_emb = self._cache[index]['task_emb']
        else:
            actions_all = np.stack(pd.read_parquet(os.path.join(self.dataset_dir, entry["action"]))['action'].values)
            task_emb = torch.load(os.path.join(self.dataset_dir, entry["task_embed"]), map_location="cpu")
            
        end_idx = min(start_frame + self.num_track_ts, len(actions_all))
        actions = torch.from_numpy(actions_all[start_frame:end_idx]).float()

        # --- 3. 读取 Tracks 和 Vis ---
        if self.cache_all and self.cache_track:
            tracks_all = self._cache[index]['tracks']
            vis_all = self._cache[index]['vis']
        else:
            npz_data = np.load(os.path.join(self.dataset_dir, entry["track"]))
            tracks_all = npz_data['tracks']
            vis_all = npz_data['vis']
            
        tracks = torch.from_numpy(tracks_all[start_frame:end_idx]).float()
        vis = torch.from_numpy(vis_all[start_frame:end_idx]).float()

        # --- 4. 尾部 Padding ---
        pad_len = self.num_track_ts - len(tracks)
        if pad_len > 0:
            tracks = torch.cat([tracks, tracks[-1:].repeat(pad_len, 1, 1)], dim=0)
            vis = torch.cat([vis, vis[-1:].repeat(pad_len, 1)], dim=0)
            
            zero_action = torch.zeros_like(actions[-1:])
            actions = torch.cat([actions, zero_action.repeat(pad_len, 1)], dim=0)
        
        # Action维度转换
        assert actions.shape[1] % 26 == 0, "Action dimension should be a multiple of 26 (pose + gripper)."
        """
        将action维度由
            "names": [
                "left_arm_joint_1_rad",
                "left_arm_joint_2_rad",
                "left_arm_joint_3_rad",
                "left_arm_joint_4_rad",
                "left_arm_joint_5_rad",
                "left_arm_joint_6_rad",
                "left_gripper_open",
                "left_eef_pos_x_m",
                "left_eef_pos_y_m",
                "left_eef_pos_z_m",
                "left_eef_rot_euler_x_rad",
                "left_eef_rot_euler_y_rad",
                "left_eef_rot_euler_z_rad",
                "right_arm_joint_1_rad",
                "right_arm_joint_2_rad",
                "right_arm_joint_3_rad",
                "right_arm_joint_4_rad",
                "right_arm_joint_5_rad",
                "right_arm_joint_6_rad",
                "right_gripper_open",
                "right_eef_pos_x_m",
                "right_eef_pos_y_m",
                "right_eef_pos_z_m",
                "right_eef_rot_euler_x_rad",
                "right_eef_rot_euler_y_rad",
                "right_eef_rot_euler_z_rad"
            ]
        转换为[
                "left_eef_pos_x_m",
                "left_eef_pos_y_m",
                "left_eef_pos_z_m",
                "left_eef_rot_euler_x_rad",
                "left_eef_rot_euler_y_rad",
                "left_eef_rot_euler_z_rad",
                "left_gripper_open",
                "right_eef_pos_x_m",
                "right_eef_pos_y_m",
                "right_eef_pos_z_m",
                "right_eef_rot_euler_x_rad",
                "right_eef_rot_euler_y_rad",
                "right_eef_rot_euler_z_rad",
                "right_gripper_open",
        ]
        """
        actions = torch.cat([
            actions[:, 7:13],  # left eef pos + rot
            actions[:, 6:7],   # left gripper
            actions[:, 20:26], # right eef pos + rot
            actions[:, 19:20], # right gripper
        ], dim=1)
        assert actions.shape[1] == 14, "After reordering, action dimension should be 14."

        # --- 5. 数据增强与采样 ---
        if np.random.rand() < self.aug_prob:
            frames = frames.unsqueeze(0)
            tracks = tracks.unsqueeze(0).unsqueeze(0)
            frames, tracks = self.augmentor((frames / 255., tracks))
            frames = frames[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        if self.uniform_sample:
            tracks, vis = sample_tracks_nearest_to_grids(tracks, vis, self.num_track_ids)
        else:
            tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)

        return frames, tracks, vis, task_emb
