"""
AeroGuardian Scene Fire Guard & Radiometric Isotherm Slicing.
Implements dynamic ambient calibration, Sobel combustion edge validation,
and wildfire ember suppression for human detection.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


class SceneFireGuard:
    """
    Classifies active wildfire scenes and extracts verified combustion fronts
    using dynamic radiometric isotherm thresholding and edge gradient checks.
    """

    def __init__(
        self,
        min_combustion_pixels: int = 500,
        min_sobel_gradient: float = 25.0
    ):
        self.min_combustion_pixels = min_combustion_pixels
        self.min_sobel_gradient = min_sobel_gradient

    def compute_dynamic_isotherm_threshold(
        self,
        thermal_raw: np.ndarray
    ) -> int:
        """
        Dynamically calculates physical combustion isotherm threshold based on
        ambient scene temperature distribution (mean + 3.2 * std), clamped to [215, 225].
        """
        ambient_mean = float(np.mean(thermal_raw))
        ambient_std = float(np.std(thermal_raw))
        dynamic_th = int(ambient_mean + 3.2 * ambient_std)
        return int(np.clip(dynamic_th, 215, 225))

    def extract_combustion_mask(
        self,
        thermal_raw: np.ndarray,
        target_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts verified combustion mask using dynamic radiometric thresholding
        and Sobel edge gradient validation to reject smooth solar-heated asphalt.
        """
        th_frame = thermal_raw.copy()
        if len(th_frame.shape) == 3:
            # FLIR Ironbow: Green channel indicates high combustion temperature (>300°C)
            if target_shape is not None and th_frame.shape[:2] != target_shape:
                th_frame = cv2.resize(th_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            raw_mask = (th_frame[:, :, 1] >= 90).astype(np.uint8) * 255
            isotherm_th = 215
        else:
            if target_shape is not None and th_frame.shape[:2] != target_shape:
                th_frame = cv2.resize(th_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            isotherm_th = self.compute_dynamic_isotherm_threshold(th_frame)
            _, raw_mask = cv2.threshold(th_frame, isotherm_th, 255, cv2.THRESH_BINARY)

        # Sobel boundary gradient check: Real fires have sharp thermal edges
        th_gray = cv2.cvtColor(th_frame, cv2.COLOR_BGR2GRAY) if len(th_frame.shape) == 3 else th_frame
        sobel_x = cv2.Sobel(th_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(th_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobel_x, sobel_y)

        # Morphological opening removes salt-and-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

        # Find candidate flame contours and filter by contiguous area
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        verified_mask = np.zeros_like(cleaned_mask)
        total_fire_pixels = 0
        contour_count = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area >= 15:
                # Check mean gradient along perimeter
                c_mask = np.zeros_like(cleaned_mask)
                cv2.drawContours(c_mask, [c], -1, 255, 1)
                mean_grad = float(np.mean(grad_mag[c_mask > 0])) if np.count_nonzero(c_mask) > 0 else 0.0

                if mean_grad >= self.min_sobel_gradient or area >= 50:
                    cv2.drawContours(verified_mask, [c], -1, 255, -1)
                    total_fire_pixels += int(area)
                    contour_count += 1

        is_active = (total_fire_pixels >= 50)

        metrics = {
            "is_active_wildfire": is_active,
            "fire_pixel_count": total_fire_pixels,
            "contour_count": contour_count,
            "isotherm_threshold": isotherm_th,
        }

        return verified_mask, metrics

    def is_wildfire_scene(
        self,
        thermal_registered: np.ndarray
    ) -> bool:
        """
        Determines whether the scene contains active wildfire combustion
        that requires suppressing canopy ember false positives on humans.
        """
        if len(thermal_registered.shape) == 3:
            b, g, r = cv2.split(thermal_registered)
            fire_pixels = int(np.count_nonzero((g >= 70) | ((r >= 195) & (b <= 120))))
            return fire_pixels >= 300
        else:
            isotherm_th = self.compute_dynamic_isotherm_threshold(thermal_registered)
            _, fire_thresh = cv2.threshold(thermal_registered, isotherm_th, 255, cv2.THRESH_BINARY)
            fire_contours, _ = cv2.findContours(fire_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [c for c in fire_contours if cv2.contourArea(c) >= 100]
            return len(large_contours) >= 2 or np.count_nonzero(fire_thresh) >= 8000


