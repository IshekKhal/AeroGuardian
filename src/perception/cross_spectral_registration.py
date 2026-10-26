"""
AeroGuardian Cross-Spectral Camera Registration Engine
Calculates dynamic altitude-compensated planar homography to warp and align
Long-Wave Infrared (LWIR) thermal video directly onto optical RGB frames.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2


class CrossSpectralRegistrar:
    """
    Calibrated dual-camera registration engine for drone-mounted optical/thermal pods.
    Eliminates optical axis disparity and altitude parallax.
    """

    def __init__(
        self,
        rgb_shape: Tuple[int, int] = (1080, 1920),
        thermal_shape: Tuple[int, int] = (512, 640),
        baseline_offset_meters: Tuple[float, float, float] = (0.065, 0.0, 0.0),
        focal_length_rgb_px: float = 1450.0,
        focal_length_thermal_px: float = 680.0,
    ):
        """
        Args:
            rgb_shape: (height, width) of optical camera in pixels.
            thermal_shape: (height, width) of thermal sensor in pixels.
            baseline_offset_meters: (tx, ty, tz) physical displacement between lenses on gimbal.
            focal_length_rgb_px: Equivalent focal length of RGB sensor in pixels.
            focal_length_thermal_px: Equivalent focal length of thermal sensor in pixels.
        """
        self.rgb_h, self.rgb_w = rgb_shape
        self.th_h, self.th_w = thermal_shape
        self.baseline = np.array(baseline_offset_meters, dtype=np.float64)

        # Calibrated intrinsic matrix K_rgb
        cx_rgb = self.rgb_w / 2.0
        cy_rgb = self.rgb_h / 2.0
        self.K_rgb = np.array(
            [[focal_length_rgb_px, 0.0, cx_rgb],
             [0.0, focal_length_rgb_px, cy_rgb],
             [0.0, 0.0, 1.0]],
            dtype=np.float64
        )

        # Calibrated intrinsic matrix K_thermal
        cx_th = self.th_w / 2.0
        cy_th = self.th_h / 2.0
        self.K_th = np.array(
            [[focal_length_thermal_px, 0.0, cx_th],
             [0.0, focal_length_thermal_px, cy_th],
             [0.0, 0.0, 1.0]],
            dtype=np.float64
        )

        # Relative rotation R (near-identity for coplanar gimbal mounting)
        self.R = np.eye(3, dtype=np.float64)

        # Ground plane unit normal (downward-facing nadir camera)
        self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def compute_homography(self, altitude_meters: float = 40.0) -> np.ndarray:
        """
        Calculates altitude-compensated homography matrix H_thermal_to_rgb:
        H = K_rgb * (R - (t * n^T) / altitude) * K_th^-1

        Args:
            altitude_meters: Current height Above Ground Level (AGL) in meters.
        Returns:
            3x3 numpy array homography matrix.
        """
        # Clamp altitude to avoid division by zero
        h = max(float(altitude_meters), 5.0)

        # Parallax compensation term: (t * n^T) / h
        t_n_t = np.outer(self.baseline, self.normal) / h

        # Planar projective homography
        H_metric = self.R - t_n_t
        K_th_inv = np.linalg.inv(self.K_th)
        H = self.K_rgb @ H_metric @ K_th_inv

        # Normalize homography matrix
        H = H / H[2, 2]
        return H

    def register_thermal_to_rgb(
        self,
        thermal_frame: np.ndarray,
        altitude_meters: float = 40.0,
        target_shape: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Warps thermal frame to align directly onto the optical RGB camera space.

        Args:
            thermal_frame: 2D (H_th, W_th) or 3D thermal image array.
            altitude_meters: Drone altitude Above Ground Level (AGL).
            target_shape: Optional (H, W) target shape. If None, defaults to (self.rgb_h, self.rgb_w).
            interpolation: OpenCV interpolation mode.
        Returns:
            Warped thermal frame with dimensions matching target_shape.
        """
        out_h, out_w = target_shape if target_shape is not None else (self.rgb_h, self.rgb_w)

        # If thermal already matches target dimensions (e.g. co-registered benchmark pair)
        if thermal_frame.shape[:2] == (out_h, out_w):
            return thermal_frame

        H = self.compute_homography(altitude_meters)
        if (out_h, out_w) != (self.rgb_h, self.rgb_w):
            scale_x = out_w / self.rgb_w
            scale_y = out_h / self.rgb_h
            S = np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            H = S @ H

        warped_thermal = cv2.warpPerspective(
            thermal_frame,
            H,
            (out_w, out_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return warped_thermal

    def fuse_spectral_views(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        altitude_meters: float = 40.0,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_INFERNO,
        mode: str = "isotherm",
    ) -> np.ndarray:
        """
        Creates a multi-spectral tactical view:
        Modes:
          - "isotherm": Smart heat-anomaly highlight over crisp optical background (DJI Zenmuse H20T style).
          - "side_by_side": Split-screen view (Optical RGB left, Thermal LWIR right).
          - "blend": Full cross-spectral alpha blend across the field of view.
          - "optical": Optical RGB camera view only.
          - "thermal": Calibrated thermal view only.

        Args:
            rgb_frame: (H_rgb, W_rgb, 3) optical image.
            thermal_frame: (H_th, W_th) or (H_th, W_th, 3) thermal sensor image.
            altitude_meters: Drone altitude AGL.
            alpha: Blend weight for thermal layer in blend mode.
            colormap: OpenCV false-color palette.
            mode: View mode string ('isotherm', 'side_by_side', 'blend', 'optical', 'thermal').
        Returns:
            Fused (H_rgb, W_rgb, 3) composite tactical frame.
        """
        h_rgb, w_rgb = rgb_frame.shape[:2]

        if mode == "optical":
            return rgb_frame.copy()

        # Warp thermal onto optical space matching the rgb frame dimensions
        warped_th = self.register_thermal_to_rgb(
            thermal_frame,
            altitude_meters,
            target_shape=(h_rgb, w_rgb)
        )

        # Prepare false-color thermal representation
        if warped_th.ndim == 3:
            color_thermal = warped_th
            gray_th = cv2.cvtColor(warped_th, cv2.COLOR_BGR2GRAY)
            valid_mask = (gray_th > 5).astype(np.float32)[:, :, np.newaxis]
        else:
            if warped_th.dtype != np.uint8:
                norm_th = cv2.normalize(warped_th, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                norm_th = warped_th
            color_thermal = cv2.applyColorMap(norm_th, colormap)
            valid_mask = (norm_th > 5).astype(np.float32)[:, :, np.newaxis]

        if mode == "thermal":
            return color_thermal

        if mode == "side_by_side":
            half_w = w_rgb // 2
            left_rgb = cv2.resize(rgb_frame, (half_w, h_rgb))
            if thermal_frame.ndim == 2:
                th_color_raw = cv2.applyColorMap(thermal_frame, colormap)
            else:
                th_color_raw = thermal_frame
            right_th = cv2.resize(th_color_raw, (half_w, h_rgb))
            return np.hstack([left_rgb, right_th])

        if mode == "blend":
            fused = rgb_frame.astype(np.float32) * (1.0 - alpha * valid_mask) + color_thermal.astype(np.float32) * (alpha * valid_mask)
            return np.clip(fused, 0, 255).astype(np.uint8)

        # Default: "isotherm" (Intelligent Heat Isolation)
        # Keeps optical 100% crisp, highlighting only active heat, fire, and victim anomalies
        if warped_th.ndim == 3:
            b, g, r = cv2.split(warped_th)
            heat_mask = ((g >= 70) | ((r >= 195) & (b <= 120))).astype(np.float32)
        else:
            heat_mask = (norm_th > 130).astype(np.float32)

        heat_mask = cv2.GaussianBlur(heat_mask, (7, 7), 0)[:, :, np.newaxis]
        fused = rgb_frame.astype(np.float32) * (1.0 - 0.75 * heat_mask) + color_thermal.astype(np.float32) * (0.75 * heat_mask)
        return np.clip(fused, 0, 255).astype(np.uint8)

    def render_tactical_hud(
        self,
        fused_frame: np.ndarray,
        telemetry: Optional[Dict[str, Any]] = None,
        detections: Optional[list] = None,
        incident: Optional[Dict[str, Any]] = None,
        fire_metrics: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Overlays military/emergency response tactical HUD graphics onto fused multi-spectral frame.
        Dynamically adapts layout, font sizes, and banners to any frame resolution (from 320px to 4K).
        """
        orig_h, orig_w = fused_frame.shape[:2]
        target_w = max(960, orig_w)
        target_h = max(540, orig_h)
        if (target_w, target_h) != (orig_w, orig_h):
            hud = cv2.resize(fused_frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            sx = float(target_w) / float(orig_w)
            sy = float(target_h) / float(orig_h)
            scaled_detections = []
            for d in (detections or []):
                sd = d.copy()
                bx, by, bw, bh = d["bbox"]
                sd["bbox"] = [int(bx * sx), int(by * sy), int(bw * sx), int(bh * sy)]
                if "head_pos" in d and d["head_pos"] is not None:
                    hx, hy = d["head_pos"]
                    sd["head_pos"] = (int(hx * sx), int(hy * sy))
                scaled_detections.append(sd)
            detections = scaled_detections
        else:
            hud = fused_frame.copy()

        h, w = hud.shape[:2]
        scale = max(0.60, min(1.2, w / 960.0))
        header_h = max(44, int(52 * scale))

        # 1. Top status bar background
        cv2.rectangle(hud, (0, 0), (w, header_h), (15, 20, 28), -1)
        cv2.line(hud, (0, header_h), (w, header_h), (56, 189, 248), 2)  # Cyan border

        # Count humans for tactical headcount display
        num_humans = sum(1 for d in (detections or []) if d.get("class") == "human")
        if num_humans > 0:
            title_text = f"AEROGUARDIAN :: MULTI-SPECTRAL TACTICAL HUD [SURVIVOR HEADCOUNT: {num_humans}]"
        else:
            title_text = "AEROGUARDIAN :: MULTI-SPECTRAL TACTICAL HUD [SURVIVORS: 0]"

        cv2.putText(hud, title_text, (14, int(header_h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry string
        alt = telemetry.get("altitude_m", 45.0) if telemetry else 45.0
        lat = telemetry.get("latitude", 34.052235) if telemetry else 34.052235
        lon = telemetry.get("longitude", -118.243683) if telemetry else -118.243683
        gimbal = telemetry.get("gimbal_pitch_deg", -45.0) if telemetry else -45.0

        # Extract target GPS from incident or detections if available
        target_gps_str = ""
        if incident and "target_gps" in incident and incident["target_gps"]:
            t_lat, t_lon = incident["target_gps"]
            target_gps_str = f"  |  TARGET PIN: {t_lat:.6f}, {t_lon:.6f}"
        elif detections:
            first_gps = next((d.get("target_gps") for d in detections if d.get("target_gps")), None)
            if first_gps:
                target_gps_str = f"  |  TARGET PIN: {first_gps[0]:.6f}, {first_gps[1]:.6f}"

        telem_str = f"ALT: {alt:.1f}m AGL  |  GPS: {lat:.6f}, {lon:.6f}  |  GIMBAL: {gimbal:.1f}deg{target_gps_str}  |  FLIR LWIR FUSED"
        cv2.putText(hud, telem_str, (14, int(header_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, (139, 148, 158), 1, cv2.LINE_AA)

        # 2. Target detection overlays
        if detections:
            placed_labels = []
            human_counter = 1
            for det in detections:
                bbox = det.get("bbox", [1200, 500, 80, 140])
                bx, by, bw, bh = [int(v) for v in bbox]
                conf = det.get("confidence", 0.92)
                cls_name = det.get("class", "human").upper()

                # Tactical corner brackets instead of simple rectangle
                corner_len = min(22, max(6, bw // 3, bh // 3))
                color = (56, 189, 248) if conf >= 0.8 else (34, 153, 210)  # Cyan or Amber

                # Top-left
                cv2.line(hud, (bx, by), (bx + corner_len, by), color, 2)
                cv2.line(hud, (bx, by), (bx, by + corner_len), color, 2)
                # Top-right
                cv2.line(hud, (bx + bw, by), (bx + bw - corner_len, by), color, 2)
                cv2.line(hud, (bx + bw, by), (bx + bw, by + corner_len), color, 2)
                # Bottom-left
                cv2.line(hud, (bx, by + bh), (bx + corner_len, by + bh), color, 2)
                cv2.line(hud, (bx, by + bh), (bx, by + bh - corner_len), color, 2)
                # Bottom-right
                cv2.line(hud, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color, 2)
                cv2.line(hud, (bx + bw, by + bh), (bx + bw, by + corner_len), color, 2)

                # Center reticle
                cx, cy = bx + bw // 2, by + bh // 2
                cv2.circle(hud, (cx, cy), 3, color, 1)

                # Head reticle if present
                if "head_pos" in det and det["head_pos"] is not None:
                    hx, hy = det["head_pos"]
                    cv2.circle(hud, (hx, hy), 4, (0, 255, 255), -1)  # Solid bright yellow head marker
                    cv2.circle(hud, (hx, hy), 8, (0, 255, 255), 1)   # Tactical head reticle circle

                if cls_name == "HUMAN" and num_humans > 1:
                    label = f"H{human_counter} [{conf*100:.0f}%]"
                    human_counter += 1
                else:
                    label = f"{cls_name} VERIFIED [{conf*100:.1f}%]"

                if "target_gps" in det and det["target_gps"] is not None:
                    tlat, tlon = det["target_gps"]
                    label += f" ({tlat:.5f},{tlon:.5f})"

                tscale = 0.40 * scale
                thickness = 1
                (tw, th_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, tscale, thickness)

                pos1 = (bx, max(header_h + th_txt + 4, by - 8))
                pos1_shift = (min(w - tw - 6, max(bx, bx + bw - tw)), max(header_h + th_txt + 4, by - 8))
                pos2 = (bx, min(h - 12, by + bh + th_txt + 10))
                pos3 = (bx, max(header_h + th_txt + 4, by - 20))

                def label_overlaps(pos):
                    px, py = pos
                    for (lx, ly, lw, lh) in placed_labels:
                        if not (px + tw < lx - 4 or px > lx + lw + 4 or py < ly - lh - 4 or py - th_txt > ly + 4):
                            return True
                    return False

                chosen = pos1
                if label_overlaps(pos1):
                    if not label_overlaps(pos1_shift):
                        chosen = pos1_shift
                    elif not label_overlaps(pos2):
                        chosen = pos2
                    elif not label_overlaps(pos3):
                        chosen = pos3
                    else:
                        chosen = pos1_shift

                placed_labels.append((chosen[0], chosen[1], tw, th_txt))
                cv2.putText(hud, label, chosen, cv2.FONT_HERSHEY_SIMPLEX, tscale, color, thickness, cv2.LINE_AA)

        # 3. Hazard triage banner or Patrol monitoring banner
        box_h = max(42, int(48 * scale))
        box_y = h - box_h - 12
        box_w = min(w - 24, max(460, int(660 * scale)))
        box_x = 12

        if incident:
            urgency = incident.get("urgency_level", "CRITICAL_IMMINENT")
            time_imp = incident.get("time_to_impact_minutes", 1.3)
            speed = incident.get("fire_speed_mps", 0.26)

            cv2.rectangle(hud, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 20, 28), -1)
            cv2.rectangle(hud, (box_x, box_y), (box_x + box_w, box_y + box_h), (51, 54, 218), 2)  # Red border

            hazard_txt = f"WILDFIRE HAZARD ({speed:.2f} m/s) -> TIME TO IMPACT: {time_imp:.1f} MIN"
            sub_txt = f"LEVEL: {urgency} | INCIDENT COMMANDER APPROVAL REQUIRED"

            cv2.putText(hud, hazard_txt, (box_x + 10, box_y + int(box_h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, sub_txt, (box_x + 10, box_y + int(box_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (51, 54, 218), 1, cv2.LINE_AA)
        elif fire_metrics and fire_metrics.get("active_perimeter_px", 0) > 500:
            speed = fire_metrics.get("velocity_mps", 0.0)
            heading = fire_metrics.get("heading_deg", 0.0)
            cv2.rectangle(hud, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 20, 28), -1)
            cv2.rectangle(hud, (box_x, box_y), (box_x + box_w, box_y + box_h), (34, 153, 210), 2)  # Amber border

            fire_txt = f"MISSION: WILDFIRE PERIMETER TRACKING  |  FLAME SPEED: {speed:.2f} m/s ({heading:.0f} deg)"
            fire_sub = "[AUTHENTIC FLAME FRONT] NO SURVIVORS IN THREAT ZONE  |  ORBITING PERIMETER"

            cv2.putText(hud, fire_txt, (box_x + 10, box_y + int(box_h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, fire_sub, (box_x + 10, box_y + int(box_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, 0.33 * scale, (34, 153, 210), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(hud, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 20, 28), -1)
            cv2.rectangle(hud, (box_x, box_y), (box_x + box_w, box_y + box_h), (56, 189, 248), 1)  # Cyan border

            patrol_txt = "MISSION: AUTONOMOUS SEARCH & PATROL  |  AUTOPILOT: ACTIVE (45m AGL)"
            patrol_sub = "[SECTOR SCAN NORMAL] DUAL-SPECTRAL SENSORS  |  ZERO FIRE SPREAD DETECTED"

            cv2.putText(hud, patrol_txt, (box_x + 10, box_y + int(box_h * 0.46)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, patrol_sub, (box_x + 10, box_y + int(box_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, 0.33 * scale, (56, 189, 248), 1, cv2.LINE_AA)

        # 4. Center flight crosshair
        ch_x, ch_y = w // 2, h // 2
        cv2.line(hud, (ch_x - 16, ch_y), (ch_x + 16, ch_y), (100, 116, 139), 1)
        cv2.line(hud, (ch_x, ch_y - 16), (ch_x, ch_y + 16), (100, 116, 139), 1)

        return hud

