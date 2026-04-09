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
