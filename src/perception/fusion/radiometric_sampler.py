"""
AeroGuardian Radiometric Sampler.
Extracts localized thermal radiometry within bounding boxes to verify physical heat signatures.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class RadiometricSampler:
    """
    Samples thermal radiometry across candidate optical detections.
    Confirms genuine biological body heat (>35C) or vehicle combustion/exhaust.
    """

    def __init__(self, human_min_peak: float = 170.0, ambient_margin: float = 7.0):
        self.human_min_peak = human_min_peak
        self.ambient_margin = ambient_margin

    def sample_bounding_box(
        self,
        th_norm: np.ndarray,
        bbox: Tuple[int, int, int, int],
        ambient_mean: float
    ) -> Dict[str, Any]:
        """
        Samples thermal radiometry within an optical bounding box.
        Returns peak, mean, and whether thermal signature confirms physical target presence.
        """
        if th_norm.ndim == 3:
            th_norm = cv2.cvtColor(th_norm, cv2.COLOR_BGR2GRAY)

        bx, by, bw, bh = bbox
        h_im, w_im = th_norm.shape[:2]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w_im, bx + bw), min(h_im, by + bh)

        if x2 <= x1 or y2 <= y1:
            return {
                "has_confirmation": False,
                "crop_max": 0.0,
                "crop_mean": 0.0
            }

        crop_th = th_norm[y1:y2, x1:x2]
        crop_max = float(np.max(crop_th))
        crop_mean = float(np.mean(crop_th))

        has_confirmation = (crop_max >= self.human_min_peak) or (crop_mean > ambient_mean + self.ambient_margin)

        return {
            "has_confirmation": has_confirmation,
            "crop_max": crop_max,
            "crop_mean": crop_mean
        }
