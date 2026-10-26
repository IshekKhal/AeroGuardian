"""
AeroGuardian Dense Optical Flow Fire Propagation & Hazard Velocity Modeling
Measures advancing wildfire perimeter displacement, compensates for drone ego-motion,
projects pixel vectors into ground metric velocity (m/s), and calculates survivor time-to-impact and GPS geolocation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class FirePropagationModel:
    """
    Computes physical hazard velocity vectors across wildfire perimeters
    using OpenCV dense optical flow with sparse background ego-motion compensation.
    """

    def __init__(
        self,
        focal_length_px: float = 1450.0,
        sensor_width_mm: float = 6.4,
        image_width_px: int = 1920,
    ):
        self.focal_length_px = focal_length_px
        self.sensor_width_mm = sensor_width_mm
        self.image_width_px = image_width_px
        self.prev_gray_fire: Optional[np.ndarray] = None
        self.prev_full_gray: Optional[np.ndarray] = None

    def compute_ground_sampling_distance(self, altitude_m: float) -> float:
        """
        Calculates Ground Sampling Distance (GSD) in meters per pixel.
        GSD = (altitude * sensor_width_mm) / (focal_length_mm * image_width)
        Or equivalently: altitude / focal_length_px
        """
        alt = max(5.0, float(altitude_m))
        return alt / self.focal_length_px

    def extract_hazard_mask(self, thermal_norm: np.ndarray, threshold: int = 200) -> np.ndarray:
        """
        Isolates active wildfire flame front and severe thermal hotspots.
        """
        _, mask = cv2.threshold(thermal_norm, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def estimate_drone_egomotion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        hazard_mask: np.ndarray,
        mavlink_groundspeed_px: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float, bool]:
        """
        Estimates camera ego-motion caused by wind drift, turbulence, and drone hover
        by tracking sparse background features on cold, unburned terrain outside the hazard mask.
        If dense smoke obscures terrain features (< 4 valid points), falls back to MAVLink groundspeed.
        Returns (dx_camera, dy_camera, is_compensated).
        """
        if prev_gray.ndim == 3:
            prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
        if curr_gray.ndim == 3:
            curr_gray = cv2.cvtColor(curr_gray, cv2.COLOR_BGR2GRAY)

        bg_mask = cv2.bitwise_not(hazard_mask)

        # Detect salient corners on stable unburned ground
        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=12,
            mask=bg_mask
        )

        if corners is None or len(corners) < 4:
            if mavlink_groundspeed_px is not None:
                return float(mavlink_groundspeed_px[0]), float(mavlink_groundspeed_px[1]), True
            return 0.0, 0.0, False

        # Track background points via sparse Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            corners,
            None,
            winSize=(21, 21),
            maxLevel=2
        )

        valid_prev = corners[status.flatten() == 1].reshape(-1, 2)
        valid_curr = curr_pts[status.flatten() == 1].reshape(-1, 2)

        if len(valid_prev) < 4:
            if mavlink_groundspeed_px is not None:
                return float(mavlink_groundspeed_px[0]), float(mavlink_groundspeed_px[1]), True
            return 0.0, 0.0, False

        displacements = valid_curr - valid_prev
        dx_ego = float(np.median(displacements[:, 0]))
        dy_ego = float(np.median(displacements[:, 1]))

        return dx_ego, dy_ego, True

    def track_hazard_velocity(
        self,
        current_gray_frame: np.ndarray,
        hazard_mask: np.ndarray,
        altitude_m: float = 45.0,
        fps: float = 30.0,
        compensate_egomotion: bool = False,
        mavlink_groundspeed_mps: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Computes dense optical flow across the masked hazard region to determine
        the mean ground velocity and spread direction of the fire front.
        Subtracts camera ego-motion (via sparse background LK flow on cold terrain,
        or MAVLink groundspeed fallback during heavy smoke) to prevent wind drift from skewing flame speed.
        """
        # Ensure frame is single-channel 8-bit grayscale for optical flow and corner detection
        if current_gray_frame.ndim == 3:
            current_gray_frame = cv2.cvtColor(current_gray_frame, cv2.COLOR_BGR2GRAY)
        elif current_gray_frame.dtype != np.uint8:
            current_gray_frame = cv2.normalize(current_gray_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if hazard_mask.shape[:2] != current_gray_frame.shape[:2]:
            hazard_mask = cv2.resize(
                hazard_mask,
                (current_gray_frame.shape[1], current_gray_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        masked_frame = cv2.bitwise_and(current_gray_frame, current_gray_frame, mask=hazard_mask)

        if self.prev_gray_fire is None or self.prev_gray_fire.shape != masked_frame.shape:
            self.prev_gray_fire = masked_frame.copy()
            self.prev_full_gray = current_gray_frame.copy()
            return {
                "velocity_mps": 0.0,
                "heading_deg": 0.0,
                "active_perimeter_px": int(np.count_nonzero(hazard_mask)),
                "ego_motion_px": (0.0, 0.0),
                "ego_compensated": False,
            }

        # Convert MAVLink groundspeed m/s to pixel displacement per frame if provided
        gsd = self.compute_ground_sampling_distance(altitude_m)
        mav_px = None
        if mavlink_groundspeed_mps is not None:
            mav_px = (
                float(mavlink_groundspeed_mps[0] / (gsd * fps)),
                float(mavlink_groundspeed_mps[1] / (gsd * fps))
            )

        # Estimate and compensate camera ego-motion
        dx_ego, dy_ego, ego_compensated = 0.0, 0.0, False
        if compensate_egomotion and self.prev_full_gray is not None:
            dx_ego, dy_ego, ego_compensated = self.estimate_drone_egomotion(
                self.prev_full_gray, current_gray_frame, hazard_mask, mavlink_groundspeed_px=mav_px
            )

        # Farneback Optical Flow calculation inside fire mask
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray_fire,
            masked_frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        self.prev_gray_fire = masked_frame.copy()
        self.prev_full_gray = current_gray_frame.copy()

        mask_indices = hazard_mask > 0
        if not np.any(mask_indices):
            return {
                "velocity_mps": 0.0,
                "heading_deg": 0.0,
                "active_perimeter_px": 0,
                "ego_motion_px": (round(dx_ego, 2), round(dy_ego, 2)),
                "ego_compensated": ego_compensated,
            }

        u = flow[..., 0][mask_indices]
        v = flow[..., 1][mask_indices]

        # Subtract camera drift from dense optical flow
        if ego_compensated:
            u_corrected = u - dx_ego
            v_corrected = v - dy_ego
        else:
            u_corrected = u
            v_corrected = v

        mean_u_px = float(np.mean(u_corrected))
        mean_v_px = float(np.mean(v_corrected))

        gsd = self.compute_ground_sampling_distance(altitude_m)
        speed_mps = float(np.sqrt(mean_u_px**2 + mean_v_px**2) * gsd * fps)

        angle_rad = np.arctan2(-mean_v_px, mean_u_px)
        heading_deg = float(np.degrees(angle_rad) % 360.0)

        return {
            "velocity_mps": round(speed_mps, 2),
            "heading_deg": round(heading_deg, 1),
            "active_perimeter_px": int(np.count_nonzero(hazard_mask)),
            "ego_motion_px": (round(dx_ego, 2), round(dy_ego, 2)),
            "ego_compensated": ego_compensated,
        }

    def compute_target_gps(
        self,
        drone_lat: float,
        drone_lon: float,
        altitude_m: float,
        target_pixel_xy: Tuple[int, int],
        image_shape: Tuple[int, int],
        gimbal_pitch_deg: float = -90.0
    ) -> Tuple[float, float]:
        """
        Projects target pixel coordinates to ground metric offsets and real GPS latitude/longitude.
        Reverse-engineered from commercial aerial photogrammetry payloads (DJI / Autel).
        """
        h_im, w_im = image_shape[:2]
        cx, cy = w_im / 2.0, h_im / 2.0
        gsd = self.compute_ground_sampling_distance(altitude_m)

        # Pixel offset from optical center
        dx_px = float(target_pixel_xy[0]) - cx
        dy_px = cy - float(target_pixel_xy[1])  # Image Y goes down, North goes up

        # Metric offsets on ground plane
        dx_meters = dx_px * gsd
        dy_meters = dy_px * gsd

        # If gimbal is tilted up from nadir (-90 deg), compensate for forward projection
        if gimbal_pitch_deg > -89.0:
            pitch_rad = np.radians(abs(gimbal_pitch_deg))
            pitch_rad = max(np.radians(15.0), min(np.radians(85.0), pitch_rad))
            forward_extension = altitude_m / np.tan(pitch_rad)
            dy_meters += forward_extension

        # Convert ground meters to GPS delta
        # 1 deg latitude ~ 111,139 meters
        d_lat = dy_meters / 111139.0
        d_lon = dx_meters / (111139.0 * np.cos(np.radians(drone_lat)))

        target_lat = float(round(drone_lat + d_lat, 6))
        target_lon = float(round(drone_lon + d_lon, 6))

        return target_lat, target_lon

    def estimate_time_to_impact(
        self,
        survivor_center_px: Tuple[int, int],
        fire_mask: np.ndarray,
        fire_velocity_mps: float,
        altitude_m: float,
    ) -> Tuple[float, float, str]:
        """
        Calculates distance from nearest fire perimeter pixel to survivor,
        and divides by velocity to compute time-to-impact in minutes.
        """
        gsd = self.compute_ground_sampling_distance(altitude_m)
        fire_coords = np.column_stack(np.where(fire_mask > 0))

        if len(fire_coords) == 0:
            return 999.0, 999.0, "LOW"

        survivor_y, survivor_x = survivor_center_px[1], survivor_center_px[0]
        distances_px = np.sqrt((fire_coords[:, 0] - survivor_y)**2 + (fire_coords[:, 1] - survivor_x)**2)
        min_dist_px = float(np.min(distances_px))
        distance_meters = round(min_dist_px * gsd, 1)

        effective_speed = max(0.2, fire_velocity_mps)
        time_seconds = distance_meters / effective_speed
        time_minutes = round(time_seconds / 60.0, 1)

        if time_minutes <= 12.0:
            urgency = "CRITICAL_IMMINENT"
        elif time_minutes <= 30.0:
            urgency = "MODERATE"
        else:
            urgency = "LOW"

        return distance_meters, time_minutes, urgency
