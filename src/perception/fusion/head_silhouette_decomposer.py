"""
AeroGuardian Biological Head-Centric Silhouette Decomposition.
Isolates individual human heads under dark tree canopies and shoulder-to-shoulder huddles.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class HeadSilhouetteDecomposer:
    """
    Biological head emissivity peak extractor for canopy search and rescue.
    Exploits maximum LWIR thermal emission (>220) from human face/scalp skin.
    """

    def __init__(
        self,
        min_cluster_width: int = 28,
        head_search_ratio: float = 0.40,
        peak_threshold: int = 180,
        min_head_separation: int = 9,
        peak_nms_distance: float = 10.0
    ):
        self.min_cluster_width = min_cluster_width
        self.head_search_ratio = head_search_ratio
        self.peak_threshold = peak_threshold
        self.min_head_separation = min_head_separation
        self.peak_nms_distance = peak_nms_distance

    def decompose(
        self,
        th_8u: np.ndarray,
        bbox: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Decomposes a candidate human thermal blob by locating biological head
        emissivity peaks in the upper silhouette. If multiple distinct heads
        are found, partitions the cluster into separate individual humans.
        """
        if th_8u.ndim == 3:
            th_8u = cv2.cvtColor(th_8u, cv2.COLOR_BGR2GRAY)

        bx, by, bw, bh = bbox
        h_im, w_im = th_8u.shape[:2]
        head_h = max(8, int(bh * self.head_search_ratio))
        y2_head = min(h_im, by + head_h)
        x2_head = min(w_im, bx + bw)
        crop_upper = th_8u[by:y2_head, bx:x2_head]

        # Guard: Single humans have width < 28; multiple shoulder-to-shoulder humans require bw >= 28
        if crop_upper.size == 0 or bw < self.min_cluster_width:
            hx = bx + bw // 2
            hy = by + max(4, int(bh * 0.15))
            if crop_upper.size > 0:
                y_off, x_off = np.unravel_index(np.argmax(crop_upper), crop_upper.shape)
                hx, hy = bx + int(x_off), by + int(y_off)
            return [{
                "class": "human",
                "confidence": 0.68,
                "bbox": bbox,
                "head_pos": (hx, hy),
                "source": "thermal_canopy_penetration"
            }]

        blurred = cv2.GaussianBlur(crop_upper, (3, 3), 0)
        col_max = np.max(blurred, axis=0)

        peaks = []
        for c in range(1, len(col_max) - 1):
            if (
                col_max[c] >= self.peak_threshold
                and col_max[c] >= col_max[c - 1]
                and col_max[c] >= col_max[c + 1]
            ):
                y_peak = int(np.argmax(blurred[:, c]))
                if not any(abs(c - p[0]) < self.min_head_separation for p in peaks):
                    peaks.append((c, y_peak, int(col_max[c])))

        if len(peaks) > 1 and bw >= self.min_cluster_width:
            humans = []
            body_w = max(14, int(bw / len(peaks) * 1.15))
            for (head_x, head_y, peak_val) in peaks:
                abs_hx = bx + head_x
                abs_hy = by + head_y
                hx1 = max(0, abs_hx - body_w // 2)
                hy1 = max(0, abs_hy - 4)
                hb = [hx1, hy1, body_w, bh]
                humans.append({
                    "class": "human",
                    "confidence": round(min(0.90, peak_val / 255.0 * 0.85), 3),
                    "bbox": hb,
                    "head_pos": (abs_hx, abs_hy),
                    "source": "head_centric_decomposition"
                })
            return humans
        else:
            hx = bx + bw // 2
            hy = by + max(4, int(bh * 0.15))
            return [{
                "class": "human",
                "confidence": 0.68,
                "bbox": bbox,
                "head_pos": (hx, hy),
                "source": "thermal_canopy_penetration"
            }]
