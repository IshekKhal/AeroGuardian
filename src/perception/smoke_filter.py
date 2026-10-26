"""
AeroGuardian Smoke Penetration & Adaptive Contrast Enhancement Filter
Monitors atmospheric visibility, detects gray smoke occlusion, and applies
Contrast Limited Adaptive Histogram Equalization (CLAHE) + bilateral filtering.
"""

from typing import Tuple
import numpy as np
import cv2


class AdaptiveSmokeFilter:
    """
    Electronic haze and smoke penetration filter for aerial search cameras.
    Detects smoke density and dynamically balances contrast.
    """

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        smoke_entropy_threshold: float = 4.8,
        min_gradient_threshold: float = 12.0,
    ):
        """
        Args:
            clip_limit: Threshold for contrast limiting in CLAHE.
            tile_grid_size: Size of local neighborhood grid for adaptive equalization.
            smoke_entropy_threshold: Shannon entropy below which image is considered smoke-blanketed.
            min_gradient_threshold: Sobel gradient magnitude below which scene is low-contrast.
        """
        self.clahe_luminance = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.clahe_thermal = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(16, 16))
        self.entropy_threshold = smoke_entropy_threshold
        self.gradient_threshold = min_gradient_threshold

    def calculate_scene_visibility(self, image: np.ndarray) -> Tuple[float, float, bool]:
        """
        Calculates image information entropy and edge gradient density.

        Returns:
            Tuple of:
            - entropy: Shannon entropy in bits (lower = more uniform/smoke-obscured).
            - mean_gradient: Mean edge magnitude (lower = hazier/blurrier).
            - is_smoke_occluded: Boolean True if visibility is degraded.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Shannon Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / (hist.sum() + 1e-7)
        non_zero = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(non_zero * np.log2(non_zero)))

        # 2. Sobel Edge Gradient Magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mean_gradient = float(np.mean(np.sqrt(gx**2 + gy**2)))

        # Smoke or fog causes both flat pixel distribution (low entropy)
        # and loss of edge high-frequency detail (low gradient)
        is_smoke = (entropy < self.entropy_threshold) and (mean_gradient < self.gradient_threshold)
        return entropy, mean_gradient, is_smoke

    def enhance_optical_smoke(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Enhances an RGB optical frame penetrating through smoke/haze:
        Converts to LAB color space, applies CLAHE to the L-channel, and merges back.
        """
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe_luminance.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def enhance_thermal_contrast(self, thermal_frame: np.ndarray) -> np.ndarray:
        """
        Maximizes thermal dynamic range across subtle heat gradients (e.g. body heat vs tree canopy).
        """
        if thermal_frame.ndim == 3:
            lab = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self.clahe_thermal.apply(l)
            enhanced = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        if thermal_frame.dtype != np.uint8:
            norm = cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            norm = thermal_frame
        return self.clahe_thermal.apply(norm)
