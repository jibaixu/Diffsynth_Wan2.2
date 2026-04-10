import torch
import torch.nn as nn
from typing import Optional


class WanVideoTrackEncoder(torch.nn.Module):
    def __init__(
        self,
        track_dim: int = 512,
        dim: int = 1536,
        num_track_per_chunk: Optional[int] = None,
        in_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
    ):
        super().__init__()
        self.track_dim = track_dim
        self.dim = dim

        if in_features is None:
            in_features = track_dim if num_track_per_chunk is None else track_dim * num_track_per_chunk
        if hidden_features is None:
            hidden_features = dim * 4 if num_track_per_chunk is not None else dim

        self.action_embedding = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_features, dim),
        )

    def forward(self, track: torch.Tensor) -> torch.Tensor:
        return self.action_embedding(track)


class WanVideoActionTrackFuser(torch.nn.Module):
    def __init__(
        self,
        dim: int = 1536,
        input_dim: Optional[int] = None,
        hidden_features: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_dim = dim * 2 if input_dim is None else input_dim
        if hidden_features is None:
            hidden_features = dim * 4

        self.fusion = nn.Sequential(
            nn.Linear(self.input_dim, hidden_features),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_features, dim),
        )

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        return self.fusion(fused_embedding)


def split_legacy_wan_action_encoder_state_dict(
    legacy_state_dict: dict[str, torch.Tensor],
    action_encoder: torch.nn.Module,
    track_encoder: torch.nn.Module,
    action_injection_mode: str,
):
    first_layer_key = "action_embedding.0.weight"
    if first_layer_key not in legacy_state_dict:
        return None, None, None

    action_linear = getattr(action_encoder.action_embedding, "0", action_encoder.action_embedding[0])
    track_linear = getattr(track_encoder.action_embedding, "0", track_encoder.action_embedding[0])
    action_in_features = int(action_linear.in_features)
    track_in_features = int(track_linear.in_features)
    legacy_weight = legacy_state_dict[first_layer_key]
    legacy_in_features = int(legacy_weight.shape[1])

    if legacy_in_features == action_in_features:
        return None, None, None

    if action_injection_mode == "adaln":
        action_dim = int(getattr(action_encoder, "action_dim", 0))
        track_dim = int(getattr(track_encoder, "track_dim", 0))
        if action_dim <= 0 or track_dim <= 0:
            return None, None, None
        if action_in_features % action_dim != 0 or track_in_features % track_dim != 0:
            return None, None, None
        num_action_chunks = action_in_features // action_dim
        num_track_chunks = track_in_features // track_dim
        if num_action_chunks != num_track_chunks:
            return None, None, None

        chunk_width = action_dim + track_dim
        expected_legacy_in_features = num_action_chunks * chunk_width
        if legacy_in_features != expected_legacy_in_features:
            return None, None, None

        action_indices = []
        track_indices = []
        for chunk_idx in range(num_action_chunks):
            chunk_start = chunk_idx * chunk_width
            action_indices.extend(range(chunk_start, chunk_start + action_dim))
            track_start = chunk_start + action_dim
            track_indices.extend(range(track_start, track_start + track_dim))
        action_index_tensor = torch.tensor(action_indices, device=legacy_weight.device)
        track_index_tensor = torch.tensor(track_indices, device=legacy_weight.device)
        action_input_weight = legacy_weight.index_select(1, action_index_tensor)
        track_input_weight = legacy_weight.index_select(1, track_index_tensor)
    else:
        expected_legacy_in_features = action_in_features + track_in_features
        if legacy_in_features != expected_legacy_in_features:
            return None, None, None
        action_input_weight = legacy_weight[:, :action_in_features]
        track_input_weight = legacy_weight[:, action_in_features:expected_legacy_in_features]

    action_state_dict = {}
    track_state_dict = {}
    for key, value in legacy_state_dict.items():
        if key == first_layer_key:
            action_state_dict[key] = action_input_weight
            track_state_dict[key] = track_input_weight
        else:
            action_state_dict[key] = value
            track_state_dict[key] = value

    message = (
        "Legacy joint action_encoder checkpoint detected; split its first-layer input "
        "weights into separate action_encoder/track_encoder initializations."
    )
    return action_state_dict, track_state_dict, message


def extract_action_only_from_legacy_wan_action_encoder_state_dict(
    legacy_state_dict: dict[str, torch.Tensor],
    action_encoder: torch.nn.Module,
    action_injection_mode: str,
):
    first_layer_key = "action_embedding.0.weight"
    if first_layer_key not in legacy_state_dict:
        return None, None

    action_linear = getattr(action_encoder.action_embedding, "0", action_encoder.action_embedding[0])
    action_in_features = int(action_linear.in_features)
    legacy_weight = legacy_state_dict[first_layer_key]
    legacy_in_features = int(legacy_weight.shape[1])
    if legacy_in_features == action_in_features or legacy_in_features < action_in_features:
        return None, None

    if action_injection_mode == "adaln":
        action_dim = int(getattr(action_encoder, "action_dim", 0))
        if action_dim <= 0 or action_in_features % action_dim != 0:
            return None, None
        num_action_chunks = action_in_features // action_dim
        if num_action_chunks <= 0 or legacy_in_features % num_action_chunks != 0:
            return None, None
        chunk_width = legacy_in_features // num_action_chunks
        if chunk_width < action_dim:
            return None, None

        action_indices = []
        for chunk_idx in range(num_action_chunks):
            chunk_start = chunk_idx * chunk_width
            action_indices.extend(range(chunk_start, chunk_start + action_dim))
        action_index_tensor = torch.tensor(action_indices, device=legacy_weight.device)
        action_input_weight = legacy_weight.index_select(1, action_index_tensor)
    else:
        action_input_weight = legacy_weight[:, :action_in_features]

    action_state_dict = {}
    for key, value in legacy_state_dict.items():
        action_state_dict[key] = action_input_weight if key == first_layer_key else value

    message = (
        "Legacy joint action_encoder checkpoint detected; extracted the action slice "
        "to initialize the baseline action_encoder."
    )
    return action_state_dict, message
