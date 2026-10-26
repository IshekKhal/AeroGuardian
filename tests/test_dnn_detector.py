"""
Unit tests for AeroGuardian OpenCV 5.0 DNN Multi-Spectral Target Detector.
"""

import pytest
import numpy as np
import cv2

from src.perception.dnn_detector import AeroTargetDetector


def test_detector_initialization():
    detector = AeroTargetDetector(model_path="models/nanodet.onnx")
    assert detector.has_dnn is True
    assert detector.net is not None
    assert len(detector.anchors_mlvl) == 3


def test_detector_thermal_hotspots():
    detector = AeroTargetDetector(model_path="models/nanodet.onnx")

    # Create synthetic thermal frame with hot spot
    thermal = np.full((512, 640), 40, dtype=np.uint8)
    cv2.circle(thermal, (320, 256), 18, 245, -1)  # Heat spot

    spots = detector.detect_thermal_hotspots(thermal, target_shape=(1080, 1920))
    assert len(spots) >= 1
    assert spots[0]["class"] == "thermal_hotspot"
    assert spots[0]["confidence"] > 0.85


def test_detector_multispectral_fusion():
    detector = AeroTargetDetector(model_path="models/nanodet.onnx")

    # Test synthetic image with no optical illumination
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    thermal = np.full((1080, 1920), 40, dtype=np.uint8)
    # Hotspot signature in the dark
    cv2.rectangle(thermal, (900, 500), (960, 620), 245, -1)

    fused = detector.detect_multispectral(rgb, thermal)
    assert len(fused) >= 1
    # Human silhouette penetration under optical occlusion/darkness
    assert fused[0]["class"] == "human"
    assert fused[0]["source"] == "thermal_canopy_penetration"
    assert 0.50 <= fused[0]["confidence"] <= 0.70


def test_detector_on_real_benchmark_images():
    detector = AeroTargetDetector(model_path="models/nanodet.onnx")
    rgb = cv2.imread("data/samples/visible/running.jpg")
    th = cv2.imread("data/samples/thermal/running.jpg", cv2.IMREAD_GRAYSCALE)
    assert rgb is not None and th is not None

    dets = detector.detect_multispectral(rgb, th)
    assert len(dets) >= 1
    classes = [d["class"] for d in dets]
    assert "human" in classes or "car" in classes

