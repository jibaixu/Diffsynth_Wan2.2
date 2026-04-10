import torch, torchvision, imageio, os, math
import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        if isinstance(data, dict):
            data = data.get("data")
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1, resize_mode="fit",):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.resize_mode = resize_mode

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        if self.resize_mode == "crop":
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height * scale), round(width * scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
            return image
        if self.resize_mode == "fit":
            target_area = target_height * target_width

            def round_by_factor(value, factor):
                return int(round(value / factor)) * factor

            def floor_by_factor(value, factor):
                return int(math.floor(value / factor)) * factor

            h_factor = max(1, int(self.height_division_factor))
            w_factor = max(1, int(self.width_division_factor))

            new_height = max(h_factor, round_by_factor(height, h_factor))
            new_width = max(w_factor, round_by_factor(width, w_factor))

            if new_height * new_width > target_area:
                beta = math.sqrt((height * width) / target_area)
                new_height = max(h_factor, floor_by_factor(height / beta, h_factor))
                new_width = max(w_factor, floor_by_factor(width / beta, w_factor))

            image = torchvision.transforms.functional.resize(
                image,
                (new_height, new_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            return image

        raise ValueError(f"Unknown resize_mode: {self.resize_mode}")
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class ToVideoTensor(DataProcessingOperator):
    """Convert loaded video frames to float tensor in (V, C, T, H, W), range [-1, 1]."""

    @staticmethod
    def _frame_to_tensor(frame: Image.Image) -> torch.Tensor:
        if not isinstance(frame, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(frame).__name__}")
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3:
            raise ValueError(f"Expected HWC frame array, got shape {array.shape}")
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # (C, H, W)
        tensor = tensor * (2.0 / 255.0) - 1.0
        return tensor

    def _frames_to_video_tensor(self, frames) -> torch.Tensor:
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("Expected non-empty frame list.")
        frame_tensors = [self._frame_to_tensor(frame) for frame in frames]
        video = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)
        return video

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            if data.ndim != 5:
                raise ValueError(f"Expected video tensor with shape (V,C,T,H,W), got {tuple(data.shape)}")
            return data.to(dtype=torch.float32)

        if isinstance(data, Image.Image):
            data = [data]

        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise TypeError("Expected loaded video frames as list/tuple.")

        if isinstance(data[0], (list, tuple)):
            views = [self._frames_to_video_tensor(view) for view in data]
            return torch.stack(views, dim=0)  # (V, C, T, H, W)

        video = self._frames_to_video_tensor(data).unsqueeze(0)  # (1, C, T, H, W)
        return video


class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _resolve_video_info(self, data, start_frame, end_frame, frame_indices):
        if isinstance(data, dict):
            path = data.get("data") 
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")
        else:
            path = data
        if not path:
            raise KeyError("Missing video path in metadata 'data' field.")

        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        else:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        return path, start_frame, end_frame, frame_indices

    def __call__(self, data: str, start_frame=None, end_frame=None, frame_indices=None):
        path, start_frame, end_frame, frame_indices = self._resolve_video_info(
            data, start_frame, end_frame, frame_indices
        )
        reader = imageio.get_reader(path)
        frames = []
        if frame_indices is None:
            num_frames = self.get_num_frames(end_frame - start_frame + 1)
            frame_indices = range(start_frame, start_frame + num_frames)
        for frame_id in frame_indices:
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _resolve_gif_info(self, data, start_frame, end_frame, frame_indices):
        if isinstance(data, dict):
            path = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")
        else:
            path = data

        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        else:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        return path, start_frame, end_frame, frame_indices

    def __call__(self, data: str, start_frame=None, end_frame=None, frame_indices=None):
        path, start_frame, end_frame, frame_indices = self._resolve_gif_info(
            data, start_frame, end_frame, frame_indices
        )
        images = iio.imread(path, mode="RGB")
        frames = []
        if frame_indices is None:
            num_frames = self.get_num_frames(end_frame - start_frame + 1)
            frame_indices = range(start_frame, start_frame + num_frames)
        for frame_id in frame_indices:
            img = images[frame_id]
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        path = data
        if isinstance(data, dict):
            path = data.get("data") 
        file_ext_name = path.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        if isinstance(data, dict):
            path = data.get("data")
            if path is None:
                return data
            if os.path.isabs(path):
                abs_path = path
            else:
                abs_path = os.path.join(self.base_path, path)
            updated = data.copy()
            updated["data"] = abs_path
            return updated
        return os.path.join(self.base_path, data)


class ResolvePromptEmbPath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        if isinstance(data, dict):
            path = data.get("data")
            if path is None:
                return data
        else:
            path = data
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)


class LoadCotrackerTrack(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        num_points_per_view=256,
    ):
        self.base_path = base_path
        self.num_frames = int(num_frames)
        self.time_division_factor = int(time_division_factor)
        self.time_division_remainder = int(time_division_remainder)
        self.num_points_per_view = int(num_points_per_view)

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def _extract_track_items(self, data):
        default_start_frame = None
        default_end_frame = None
        default_frame_indices = None
        if isinstance(data, dict) and "data" not in data:
            default_start_frame = data.get("start_frame")
            default_end_frame = data.get("end_frame")
            default_frame_indices = data.get("frame_indices")
            data = data.get("track")

        if data is None:
            return None, default_start_frame, default_end_frame, default_frame_indices

        if isinstance(data, (str, os.PathLike, dict)):
            data = [data]
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise TypeError("Expected track metadata as a non-empty list of paths.")
        return list(data), default_start_frame, default_end_frame, default_frame_indices

    def _resolve_track_info(self, data, default_start_frame=None, default_end_frame=None, default_frame_indices=None):
        if isinstance(data, dict):
            track_rel = data.get("data")
            start_frame = data.get("start_frame", default_start_frame)
            end_frame = data.get("end_frame", default_end_frame)
            frame_indices = data.get("frame_indices", default_frame_indices)
        else:
            track_rel = data
            start_frame = default_start_frame
            end_frame = default_end_frame
            frame_indices = default_frame_indices

        if track_rel is None:
            return None, start_frame, end_frame, frame_indices

        if isinstance(track_rel, os.PathLike):
            track_rel = os.fspath(track_rel)
        if not isinstance(track_rel, str) or not track_rel:
            raise TypeError(f"Unexpected track path payload: {track_rel!r}")

        if os.path.isabs(track_rel):
            track_path = track_rel
        else:
            track_path = os.path.join(self.base_path, track_rel)
        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        return track_path, start_frame, end_frame, frame_indices

    def _select_frame_indices(self, total_frames, start_frame, end_frame, frame_indices):
        if frame_indices is not None:
            indices = [int(frame_id) for frame_id in frame_indices]
        else:
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = total_frames - 1
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            num_frames = self.get_num_frames(end_frame - start_frame + 1)
            indices = list(range(start_frame, start_frame + num_frames))

        if len(indices) == 0:
            raise ValueError("Track frame selection is empty.")
        invalid = [frame_id for frame_id in indices if frame_id < 0 or frame_id >= total_frames]
        if invalid:
            raise IndexError(f"Track frame indices out of range: {invalid[:8]} (total_frames={total_frames})")
        return indices

    def _sample_point_indices(self, visible_mask: np.ndarray, total_points: int) -> np.ndarray:
        visible_ids = np.flatnonzero(visible_mask.astype(bool))
        if len(visible_ids) == 0:
            visible_ids = np.arange(total_points, dtype=np.int64)
        if len(visible_ids) >= self.num_points_per_view:
            return np.random.choice(visible_ids, size=self.num_points_per_view, replace=False)

        base = np.random.permutation(visible_ids)
        remaining = self.num_points_per_view - len(base)
        if len(base) == 0:
            pad_source = np.arange(total_points, dtype=np.int64)
        else:
            pad_source = base
        pad = np.random.choice(pad_source, size=remaining, replace=True)
        return np.concatenate([base, pad], axis=0)

    def __call__(self, data, start_frame=None, end_frame=None, frame_indices=None):
        track_items, default_start_frame, default_end_frame, default_frame_indices = self._extract_track_items(data)
        if track_items is None:
            return None

        if start_frame is not None:
            default_start_frame = start_frame
        if end_frame is not None:
            default_end_frame = end_frame
        if frame_indices is not None:
            default_frame_indices = frame_indices

        views = []
        for item in track_items:
            track_path, item_start_frame, item_end_frame, item_frame_indices = self._resolve_track_info(
                item,
                default_start_frame=default_start_frame,
                default_end_frame=default_end_frame,
                default_frame_indices=default_frame_indices,
            )
            if track_path is None:
                continue

            with np.load(track_path) as track_file:
                if "tracks" not in track_file or "vis" not in track_file:
                    raise KeyError(f"Track file must contain `tracks` and `vis`: {track_path}")
                tracks = np.asarray(track_file["tracks"], dtype=np.float32)
                vis = np.asarray(track_file["vis"], dtype=bool)

            if tracks.ndim != 3 or tracks.shape[-1] != 2:
                raise ValueError(f"Unexpected track shape {tracks.shape} in {track_path}")
            if vis.ndim != 2 or vis.shape != tracks.shape[:2]:
                raise ValueError(f"Unexpected visibility shape {vis.shape} for tracks {tracks.shape} in {track_path}")

            selected_frame_indices = self._select_frame_indices(
                tracks.shape[0],
                item_start_frame,
                item_end_frame,
                item_frame_indices,
            )
            sampled_tracks = tracks[selected_frame_indices]
            sampled_vis = vis[selected_frame_indices]
            point_indices = self._sample_point_indices(sampled_vis[0], sampled_tracks.shape[1])
            views.append(sampled_tracks[:, point_indices, :])

        if len(views) == 0:
            return None

        combined = np.concatenate(views, axis=1)
        return combined[None, ...].astype(np.float32, copy=False)


OBS_ACTION_NAMES = [
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
    "right_eef_rot_euler_z_rad",
]

JOINT_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
]

POSE_NAMES = [
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


class LoadCobotAction(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        action_type="state_joint",
        stat=None,
        use_percentile_stats=True,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        if action_type not in ("state_joint", "state_pose", "action_joint", "action_pose"):
            raise ValueError(f"Unsupported action type: {action_type}")
        self.base_path = base_path
        self.action_type = action_type
        self.stat = stat or {}
        self.use_percentile_stats = use_percentile_stats
        self.use_state = action_type.startswith("state_")
        self.use_joint = action_type.endswith("_joint")
        name_to_idx = {name: idx for idx, name in enumerate(OBS_ACTION_NAMES)}
        self.indices = [name_to_idx[name] for name in (JOINT_NAMES if self.use_joint else POSE_NAMES)]
        self._stat_min = None
        self._stat_max = None
        if self.stat and action_type in self.stat:
            entry = self.stat[action_type]
            if self.use_percentile_stats:
                self._stat_min = np.asarray(entry.get("p01", []), dtype=np.float32)
                self._stat_max = np.asarray(entry.get("p99", []), dtype=np.float32)
            else:
                self._stat_min = np.asarray(entry.get("min", []), dtype=np.float32)
                self._stat_max = np.asarray(entry.get("max", []), dtype=np.float32)

    def _resolve_parquet_info(self, data, start_frame, end_frame, frame_indices):
        if isinstance(data, dict):
            parquet_rel = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
            if frame_indices is None:
                frame_indices = data.get("frame_indices")
        else:
            parquet_rel = data
        if not parquet_rel:
            raise KeyError("Missing parquet path in metadata 'data' field.")
        if os.path.isabs(parquet_rel):
            parquet_path = parquet_rel
        else:
            parquet_path = os.path.join(self.base_path, parquet_rel)

        if frame_indices is not None:
            frame_indices = [int(frame_id) for frame_id in frame_indices]
        else:
            start_frame = int(start_frame)
            end_frame = int(end_frame)
        return parquet_path, start_frame, end_frame, frame_indices

    def _get_min_max(self):
        if self._stat_min is not None and self._stat_max is not None:
            return self._stat_min, self._stat_max
        raise KeyError(f"Missing normalization stats for action type: {self.action_type}")

    def _normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1.0
        return np.clip(ndata, clip_min, clip_max)

    def _read_slice(self, parquet_path, column, start_frame, num_frames):
        start = int(start_frame)
        end = start + int(num_frames)
        table = pq.read_table(parquet_path, columns=[column])
        data = table.to_pydict()[column]
        if end > len(data):
            raise ValueError(
                f"Not enough rows in {parquet_path} for slice "
                f"start={start_frame}, num_frames={num_frames}"
            )
        return np.asarray(data[start:end], dtype=np.float32)

    def _read_indices(self, parquet_path, column, frame_indices):
        table = pq.read_table(parquet_path, columns=[column])
        data = table.to_pydict()[column]
        values = [data[int(frame_id)] for frame_id in frame_indices]
        return np.asarray(values, dtype=np.float32)

    def get_num_frames(self, total_frames):
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, data: str, start_frame=None, end_frame=None, frame_indices=None):
        parquet_path, start_frame, end_frame, frame_indices = self._resolve_parquet_info(
            data, start_frame, end_frame, frame_indices
        )
        column = "observation.state" if self.use_state else "action"
        if frame_indices is None:
            num_frames = self.get_num_frames(end_frame - start_frame + 1)
            arr = self._read_slice(parquet_path, column, start_frame, num_frames)
        else:
            arr = self._read_indices(parquet_path, column, frame_indices)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected action shape {arr.shape} in {parquet_path}")
        if arr.shape[1] == len(OBS_ACTION_NAMES):
            arr = arr[:, self.indices]
        elif self.use_joint and arr.shape[1] == len(JOINT_NAMES):
            pass
        elif (not self.use_joint) and arr.shape[1] == len(POSE_NAMES):
            pass
        else:
            raise ValueError(
                f"Unexpected action width {arr.shape[1]} for action type {self.action_type} in {parquet_path}"
            )
        min_vals, max_vals = self._get_min_max()
        arr = self._normalize_bound(arr, min_vals, max_vals)
        return arr[None, ...]
