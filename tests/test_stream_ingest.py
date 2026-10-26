"""
Unit tests for DualStreamIngest video and benchmark dataset ingestion pipeline.
"""

import pytest
import numpy as np

from src.perception.stream_ingest import DualStreamIngest


def test_dual_stream_dataset_discovery():
    ingest = DualStreamIngest(samples_dir="data/samples")
    assert ingest.has_real_dataset_pairs() is True
    assert len(ingest.paired_files) >= 4

    rgb, thermal, meta = ingest.load_dataset_pair(idx=0)
    assert rgb.ndim == 3
    assert thermal.ndim == 2
    assert rgb.dtype == np.uint8
    assert thermal.dtype == np.uint8
    assert "fire_mask" in meta
    assert meta["source"] == "authentic_benchmark_dataset"


def test_dual_stream_generator_loop():
    ingest = DualStreamIngest(samples_dir="data/samples")
    frames = list(ingest.stream_frames(max_frames=4, throttle=False))

    assert len(frames) == 4
    for idx, (frame_idx, rgb, thermal, meta) in enumerate(frames):
        assert frame_idx == idx
        assert rgb.ndim == 3
        assert thermal.ndim == 2
        assert "timestamp" in meta
        assert "sequence_name" in meta
        assert "telemetry" in meta
        assert "latitude" in meta["telemetry"]
        assert "longitude" in meta["telemetry"]


def test_dji_srt_telemetry_generation_and_parsing():
    from src.perception.stream_ingest import generate_dji_srt_entry, parse_dji_srt
    srt_str = generate_dji_srt_entry(
        frame_idx=1,
        start_sec=0.0,
        duration_sec=0.033,
        lat=34.052235,
        lon=-118.243683,
        alt_rel=45.0,
        pitch=-45.0,
        yaw=12.4
    )
    assert "00:00:00,000 --> 00:00:00,033" in srt_str
    assert "[latitude: 34.052235]" in srt_str
    assert "[pitch: -45.0]" in srt_str

    records = parse_dji_srt(srt_str)
    assert len(records) == 1
    rec = records[0]
    assert rec["frame_idx"] == 1
    assert abs(rec["latitude"] - 34.052235) < 1e-5
    assert abs(rec["longitude"] - (-118.243683)) < 1e-5
    assert rec["altitude_m"] == 45.0
    assert rec["gimbal_pitch_deg"] == -45.0
    assert rec["gimbal_yaw_deg"] == 12.4


def test_continuous_temporal_stream_optical_flow():
    from src.hazard.fire_flow_model import FirePropagationModel
    ingest = DualStreamIngest(samples_dir="data/samples")
    # Find wildfire index dynamically
    wildfire_idx = next((i for i, f in enumerate(ingest.paired_files) if "wildfire" in f[0].lower()), 0)
    frames = list(ingest.stream_temporal_sequence(base_pair_idx=wildfire_idx, num_frames=5, flame_drift_px=(1.2, -0.6)))
    assert len(frames) == 5

    hazard_model = FirePropagationModel()
    _, rgb0, th0, meta0 = frames[0]
    _, rgb1, th1, meta1 = frames[1]

    # Flow should calculate frame-to-frame progression smoothly
    flow0 = hazard_model.track_hazard_velocity(th0, meta0["fire_mask"], altitude_m=45.0)
    flow1 = hazard_model.track_hazard_velocity(th1, meta1["fire_mask"], altitude_m=45.0)
    assert flow1["velocity_mps"] > 0.1


def test_continuous_video_file_stream():
    """Validates real 15-second dual-stream MP4 video files with companion SRT telemetry."""
    ingest = DualStreamIngest(samples_dir="data/samples")
    assert ingest.has_real_videos() is True

    video_frames = list(ingest.stream_continuous_video(max_frames=15, throttle=False))
    assert len(video_frames) == 15

    for idx, (f_idx, rgb, th, meta) in enumerate(video_frames):
        assert f_idx == idx
        assert rgb.shape == (720, 1280, 3)
        assert th.shape == (512, 640)
        assert meta["source"] == "continuous_flight_video_mp4"
        assert "telemetry" in meta
        assert "latitude" in meta["telemetry"]
        assert "gimbal_pitch_deg" in meta["telemetry"]
        assert "fire_mask" in meta


def test_flame2_multi_pair_discovery_and_streaming():
    """Validates multi-pair discovery and streaming across authentic FLAME 2 flight pairs."""
    ingest = DualStreamIngest(samples_dir="data/samples")
    pairs = ingest.get_available_pairs()
    assert len(pairs) >= 1

    # Verify active pair streaming
    pair_frames = list(ingest.stream_flame2_flight(max_frames=5, throttle=False))
    assert len(pair_frames) == 5

    for idx, (f_idx, rgb, th, meta) in enumerate(pair_frames):
        assert f_idx == idx
        # Optical should be 1080p (or resized down from 4K)
        assert rgb.ndim == 3
        assert rgb.shape[1] <= 1920
        # Thermal should be present
        assert th.shape[:2] == (512, 640)
        assert "pair_id" in meta
        assert "telemetry" in meta
        assert abs(meta["telemetry"]["latitude"] - 35.031500) < 0.01
        assert abs(meta["telemetry"]["longitude"] - (-111.597800)) < 0.01


