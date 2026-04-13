import unittest

import torch

from diffsynth.models.wan_video_track_adapter import WanTrackResidualAdapter
from diffsynth.pipelines.wan_video import (
    WanVideoUnit_TrackContextEmbedder,
    _validate_track_residual_shape,
)


class _DummyVAE:
    upsampling_factor = 8


class _DummyPipe:
    def __init__(self):
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.track_context_adapter = WanTrackResidualAdapter(
            track_in_dim=4,
            hidden_dim=32,
            out_dim=48,
            patch_size=(1, 2, 4),
            num_layers=2,
        )
        self.track_num_points_per_view = 256
        self.num_track_views = 1
        self.track_noise_std = 0.0
        self.track_clip_min = -0.25
        self.track_clip_max = 1.25
        self.vae = _DummyVAE()

    def load_models_to_device(self, *_args, **_kwargs):
        return None


class WanVideoTrackResidualTests(unittest.TestCase):
    def test_track_context_embedder_aligns_residual_tokens_to_latent_time(self):
        pipe = _DummyPipe()
        unit = WanVideoUnit_TrackContextEmbedder()
        num_frames = 17
        height = 480
        width = 640
        expected_points = pipe.track_num_points_per_view * pipe.num_track_views
        track = torch.linspace(
            0.0,
            1.0,
            steps=num_frames * expected_points * 2,
            dtype=torch.float32,
        ).view(1, num_frames, expected_points, 2)

        outputs = unit.process(
            pipe,
            track=track,
            num_frames=num_frames,
            height=height,
            width=width,
        )

        track_residuals = outputs["track_residuals"]
        latent_frames = ((num_frames - 1) // 4) + 1
        patch_t, patch_h, patch_w = pipe.track_context_adapter.patch_size
        latent_height = height // pipe.vae.upsampling_factor
        latent_width = width // pipe.vae.upsampling_factor
        expected_tokens = (
            ((latent_frames - patch_t) // patch_t) + 1
        ) * (
            ((latent_height - patch_h) // patch_h) + 1
        ) * (
            ((latent_width - patch_w) // patch_w) + 1
        )

        self.assertEqual(expected_tokens, 3000)
        self.assertEqual(len(track_residuals), pipe.track_context_adapter.num_layers)
        for residual in track_residuals:
            self.assertEqual(tuple(residual.shape), (1, expected_tokens, pipe.track_context_adapter.out_dim))

    def test_validate_track_residual_shape_raises_before_residual_add(self):
        x = torch.zeros(1, 3000, 48)
        track_delta = torch.zeros(1, 10200, 48)

        with self.assertRaisesRegex(ValueError, "Track residual token shape mismatch"):
            _validate_track_residual_shape(
                track_delta=track_delta,
                x=x,
                block_id=15,
                residual_id=0,
                grid_size=(5, 30, 20),
            )


if __name__ == "__main__":
    unittest.main()
