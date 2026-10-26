"""
AeroGuardian Multi-Spectral Target Detector.
Coordinates optical neural detection, thermal radiometry, biological head decomposition,
and wildfire scene classification via modular submodules.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.perception.detection.nanodet_onnx import NanoDetDetector, COCO_CLASSES
from src.perception.fusion.radiometric_sampler import RadiometricSampler
from src.perception.fusion.head_silhouette_decomposer import HeadSilhouetteDecomposer
from src.perception.classifiers.scene_fire_guard import SceneFireGuard


class AeroTargetDetector:
    """
    Dual-mode target detector utilizing OpenCV 5 native DNN inference
    and thermal radiometric cross-validation.
    """

    def __init__(
        self,
        model_path: str = "models/nanodet.onnx",
        conf_threshold: float = 0.18,
        nms_threshold: float = 0.50,
        input_size: Tuple[int, int] = (416, 416),
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Modular components
        self.optical_detector = NanoDetDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size
        )
        self.radiometric_sampler = RadiometricSampler()
        self.decomposer = HeadSilhouetteDecomposer()
        self.fire_guard = SceneFireGuard()

        # Expose legacy attributes for backwards compatibility
        self.has_dnn = self.optical_detector.has_dnn
        self.net = self.optical_detector.net
        self.anchors_mlvl = self.optical_detector.anchors_mlvl

    def detect_optical(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs native OpenCV 5 DNN detection on the optical frame.
        """
        return self.optical_detector.detect(image_bgr)

    def _decompose_heads(
        self,
        th_8u: np.ndarray,
        bbox: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Decomposes candidate human thermal silhouette into distinct biological heads.
        """
        return self.decomposer.decompose(th_8u, bbox)

    def detect_thermal_hotspots(
        self,
        thermal_norm: np.ndarray,
        target_shape: Tuple[int, int] = (1080, 1920),
        temp_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts high-temperature radiometric anomalies from thermal frames.
        Corresponds to body heat signatures (>35C) or combustion borders.
        """
        th_8u = cv2.normalize(thermal_norm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if th_8u.shape[:2] != target_shape:
            th_8u = cv2.resize(th_8u, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

        if temp_threshold is None:
            mean_val = float(np.mean(th_8u))
            std_val = float(np.std(th_8u))
            computed_th = int(min(220, max(125, mean_val + 1.15 * std_val)))
        else:
            computed_th = temp_threshold

        _, thresh = cv2.threshold(th_8u, computed_th, 255, cv2.THRESH_BINARY)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_tgt, w_tgt = target_shape
        margin_x = int(w_tgt * 0.02)
        margin_y = int(h_tgt * 0.02)
        min_area = max(16, int(w_tgt * h_tgt * 0.0001))
        max_area = int(w_tgt * h_tgt * 0.25)

        hotspots = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                if x < margin_x or x + w > w_tgt - margin_x or y < margin_y or y + h > h_tgt - margin_y:
                    continue
                peak_val = float(np.max(th_8u[y:y+h, x:x+w]))
                aspect = float(h) / max(float(w), 1.0)
                if peak_val >= 195 and 0.5 <= aspect <= 4.5:
                    conf = min(0.99, peak_val / 255.0)
                    hotspots.append({
                        "class": "thermal_hotspot",
                        "confidence": round(conf, 3),
                        "bbox": [x, y, w, h],
                        "source": "thermal_lwir"
                    })

        return hotspots

    def detect_multispectral(
        self,
        image_bgr: np.ndarray,
        thermal_registered: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Fuses optical neural detections with registered thermal radiometry.
        Directly measures physical thermal emissivity within detected bounding boxes.
        Elevates confidence when true body/engine heat signatures match optical targets.
        """
        optical_dets = self.detect_optical(image_bgr)
        is_active_wildfire = self.fire_guard.is_wildfire_scene(thermal_registered)

        if thermal_registered.ndim == 3:
            th_gray = cv2.cvtColor(thermal_registered, cv2.COLOR_BGR2GRAY)
        else:
            th_gray = thermal_registered

        th_norm = cv2.normalize(th_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if th_norm.shape[:2] != image_bgr.shape[:2]:
            th_norm = cv2.resize(th_norm, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        ambient_mean = float(np.mean(th_norm))
        h_im, w_im = image_bgr.shape[:2]
        fused_detections: List[Dict[str, Any]] = []

        # 1. Evaluate optical detections against localized thermal radiometry
        is_ironbow = (thermal_registered.ndim == 3)
        for opt in optical_dets:
            bx, by, bw, bh = opt["bbox"]
            if bx < 10 or by < 10 or bx + bw > w_im - 10 or by + bh > h_im - 10:
                continue

            sample_result = self.radiometric_sampler.sample_bounding_box(
                th_norm=th_norm,
                bbox=(bx, by, bw, bh),
                ambient_mean=ambient_mean
            )
            has_thermal_confirmation = sample_result["has_confirmation"]

            if is_ironbow:
                crop_th = thermal_registered[by:by+bh, bx:bx+bw]
                if crop_th.size > 0:
                    has_ironbow_heat = bool(np.max(crop_th[:, :, 2]) >= 130)
                    is_combustion_core = bool(np.mean(crop_th[:, :, 2]) > 230 and np.mean(crop_th[:, :, 1]) > 175)
                    if is_combustion_core:
                        continue
                    has_thermal_confirmation = has_thermal_confirmation or has_ironbow_heat

            if opt["class"] == "human":
                # In active wildfire, discard detections that overlap combustion (>235) or lack thermal confirmation
                if is_active_wildfire and (sample_result["crop_max"] > 235 or not has_thermal_confirmation):
                    continue

                decomp = self.decomposer.decompose(th_norm, opt["bbox"])
                for h in decomp:
                    conf = min(0.95, max(h["confidence"], opt["confidence"] * 3.8 if has_thermal_confirmation else opt["confidence"]))
                    h["confidence"] = round(conf, 3)
                    h["source"] = "fused_dnn_thermal" if has_thermal_confirmation else "optical_dnn"
                    fused_detections.append(h)
            else:
                if has_thermal_confirmation:
                    fused_conf = min(0.95, opt["confidence"] * 1.55)
                    fused_detections.append({
                        "class": opt["class"],
                        "confidence": round(fused_conf, 3),
                        "bbox": opt["bbox"],
                        "source": "fused_dnn_thermal"
                    })
                elif opt["confidence"] >= 0.52:
                    fused_detections.append({
                        "class": opt["class"],
                        "confidence": round(opt["confidence"], 3),
                        "bbox": opt["bbox"],
                        "source": "optical_dnn"
                    })

        # 2. Check for strong isolated thermal anomalies under heavy canopy or darkness
        if not is_active_wildfire:
            thermal_spots = self.detect_thermal_hotspots(th_norm, target_shape=(h_im, w_im))
            for spot in thermal_spots:
                sx, sy, sw, sh = spot["bbox"]
                aspect = float(sh) / max(float(sw), 1.0)
                if 0.7 <= aspect <= 4.0 and 20 <= sh <= 140 and sw <= 70:
                    decomp = self.decomposer.decompose(th_norm, [sx, sy, sw, sh])
                    for d in decomp:
                        s_head = d.get("head_pos", (d["bbox"][0] + d["bbox"][2] // 2, d["bbox"][1] + d["bbox"][3] // 4))
                        already_covered = False
                        for existing in fused_detections:
                            if existing["class"] == "human":
                                e_head = existing.get("head_pos", (existing["bbox"][0] + existing["bbox"][2] // 2, existing["bbox"][1] + existing["bbox"][3] // 4))
                                if np.hypot(s_head[0] - e_head[0], s_head[1] - e_head[1]) < 20:
                                    already_covered = True
                                    break
                        if not already_covered:
                            d["confidence"] = round(min(0.68, spot["confidence"] * 0.70), 3)
                            d["source"] = "thermal_canopy_penetration"
                            fused_detections.append(d)

        # 3. Final cross-spectral overlap and containment suppression
        final_detections = []
        for d1 in sorted(fused_detections, key=lambda x: x["confidence"], reverse=True):
            duplicate = False
            if d1["class"] == "human":
                h1 = d1.get("head_pos")
                for d2 in final_detections:
                    if d2["class"] == "human":
                        h2 = d2.get("head_pos")
                        if h1 and h2 and np.hypot(h1[0] - h2[0], h1[1] - h2[1]) < 10:
                            duplicate = True
                            break
            else:
                x1, y1, w1, h1 = d1["bbox"]
                a1 = w1 * h1
                for d2 in final_detections:
                    if d1["class"] == d2["class"]:
                        x2, y2, w2, h2 = d2["bbox"]
                        ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                        iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                        inter = ix * iy
                        union = a1 + (w2 * h2) - inter
                        iou = inter / max(union, 1.0)
                        containment = inter / max(min(a1, w2 * h2), 1.0)
                        if iou >= 0.35 or containment >= 0.60:
                            duplicate = True
                            break
            if not duplicate:
                final_detections.append(d1)

        return sorted(final_detections, key=lambda x: x["bbox"][0])
