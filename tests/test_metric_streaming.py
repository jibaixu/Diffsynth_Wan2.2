import json
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

METRIC_FILE = Path(__file__).resolve().parents[1] / "diffsynth" / "core" / "metric" / "metric.py"

if "torch" not in sys.modules:
    torch_stub = types.SimpleNamespace(device=object)
    sys.modules["torch"] = torch_stub

metric_spec = importlib.util.spec_from_file_location("metric_under_test", METRIC_FILE)
metric = importlib.util.module_from_spec(metric_spec)
assert metric_spec.loader is not None
metric_spec.loader.exec_module(metric)


def _make_comparison_video(num_frames=4, num_views=2, view_height=3, view_width=4):
    frames = np.zeros((num_frames, num_views * view_height, 2 * view_width, 3), dtype=np.float32)
    for frame_idx in range(num_frames):
        for view_idx in range(num_views):
            row_start = view_idx * view_height
            row_end = row_start + view_height
            frames[frame_idx, row_start:row_end, :view_width] = 0.1 * (view_idx + 1)
            frames[frame_idx, row_start:row_end, view_width:] = 0.2 * (view_idx + 1)
    return frames


class MetricStreamingTests(unittest.TestCase):
    def test_iter_prepared_sample_batches_streams_by_video_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            comparison_dir = Path(tmp_dir)
            video_a = comparison_dir / "pick" / "video_a.mp4"
            video_b = comparison_dir / "pick" / "video_b.mp4"
            video_a.parent.mkdir(parents=True, exist_ok=True)
            video_a.touch()
            video_b.touch()

            sample_records = [
                {"video_path": str(video_a), "prompt": "prompt a"},
                {"video_path": str(video_b), "prompt": "prompt b"},
            ]
            base_ctx = metric.EvalContext(
                comparison_dir=str(comparison_dir),
                video_files=[str(video_a), str(video_b)],
                view_names=["front", "wrist"],
                num_views=2,
                frame_chunk_size=2,
                device="cpu",
                streaming_eval=True,
                eval_video_batch_size=1,
                eval_num_workers=2,
                sample_records=sample_records,
                prompt_map_jsonl=None,
            )
            prompt_lookup = metric._build_prompt_lookup(base_ctx)
            ctx = metric.EvalContext(
                comparison_dir=base_ctx.comparison_dir,
                video_files=base_ctx.video_files,
                view_names=base_ctx.view_names,
                num_views=base_ctx.num_views,
                frame_chunk_size=base_ctx.frame_chunk_size,
                device=base_ctx.device,
                streaming_eval=base_ctx.streaming_eval,
                eval_video_batch_size=base_ctx.eval_video_batch_size,
                eval_num_workers=base_ctx.eval_num_workers,
                sample_records=base_ctx.sample_records,
                prompt_map_jsonl=base_ctx.prompt_map_jsonl,
                prompt_lookup=prompt_lookup,
            )
            video_frames = {
                str(video_a): _make_comparison_video(),
                str(video_b): _make_comparison_video(),
            }

            with mock.patch.object(metric, "read_video_decord", side_effect=lambda path: video_frames[path]):
                batches = list(metric._iter_prepared_sample_batches(ctx, desc="test"))

            self.assertEqual(len(batches), 2)
            self.assertEqual([len(batch) for batch in batches], [2, 2])
            self.assertEqual(batches[0][0].prompt, "prompt a")
            self.assertEqual(batches[1][0].prompt, "prompt b")
            self.assertEqual(batches[0][0].frames, 4)
            self.assertEqual(tuple(batches[0][0].gt_video.shape), (4, 3, 4, 3))
            self.assertEqual(tuple(batches[0][0].pred_video.shape), (4, 3, 4, 3))

    def test_evaluate_and_write_report_forwards_streaming_options(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            comparison_dir = Path(tmp_dir) / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = Path(tmp_dir) / "metrics.json"
            sample_records = [
                {
                    "video_path": str(comparison_dir / "pick" / "video_a.mp4"),
                    "sample_type": "pick",
                    "prompt": "prompt a",
                }
            ]
            fake_metrics = {
                "overall": {"psnr": 1.0},
                "views": {"front": {"psnr": 1.0, "frames": 4}},
            }

            with mock.patch.object(metric, "evaluate", return_value=fake_metrics) as evaluate_mock:
                report = metric.evaluate_and_write_report(
                    comparison_dir=str(comparison_dir),
                    checkpoint_name="epoch-49.safetensors",
                    metrics_output_path=str(metrics_path),
                    sample_records=sample_records,
                    num_workers=12,
                    num_views=2,
                    frame_chunk_size=4,
                    metric_preset="core",
                    streaming_eval=False,
                    eval_video_batch_size=3,
                    eval_num_workers=5,
                )

            self.assertEqual(evaluate_mock.call_count, 1)
            for call in evaluate_mock.call_args_list:
                self.assertEqual(call.kwargs["streaming_eval"], False)
                self.assertEqual(call.kwargs["eval_video_batch_size"], 3)
                self.assertEqual(call.kwargs["eval_num_workers"], 5)
                self.assertEqual(call.kwargs["metric_preset"], "core")
            self.assertEqual(report["checkpoint"], "epoch-49.safetensors")
            self.assertNotIn("types", report)
            with metrics_path.open("r", encoding="utf-8") as file_handle:
                written_report = json.load(file_handle)
            self.assertEqual(written_report["epoch"], "epoch-49")
            self.assertNotIn("types", written_report)


if __name__ == "__main__":
    unittest.main()
