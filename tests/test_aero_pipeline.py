"""
AeroGuardian Automated Regression & Pipeline Test Suite
Verifies cross-spectral homography, adaptive smoke filtering, autopilot dispatch,
dense optical flow fire velocity modeling, and the complete Agentic decision loop.
"""

import unittest
import time
import numpy as np
import cv2

from src.perception.cross_spectral_registration import CrossSpectralRegistrar
from src.perception.smoke_filter import AdaptiveSmokeFilter
from src.flight.mavlink_interface import MAVLinkAutopilotController, TelemetryPacket
from src.hazard.fire_flow_model import FirePropagationModel
from src.agent.orchestrator import AeroGuardianOrchestrator, MissionState


class TestAeroGuardianPipeline(unittest.TestCase):

    def setUp(self):
        # Create synthetic 1080p optical test frame and 512x640 thermal frame
        self.rgb_h, self.rgb_w = 1080, 1920
        self.th_h, self.th_w = 512, 640

        # Create synthetic textured terrain for clear optical test
        np.random.seed(42)
        base_noise = np.random.randint(40, 180, (self.rgb_h, self.rgb_w, 3), dtype=np.uint8)
        self.synth_rgb = base_noise
        # Add terrain trees/rocks
        cv2.rectangle(self.synth_rgb, (200, 200), (1600, 900), (34, 139, 34), 20)
        cv2.circle(self.synth_rgb, (960, 540), 100, (80, 120, 80), -1)

        self.synth_thermal = np.full((self.th_h, self.th_w), 40, dtype=np.uint8)
        # Hotspot in thermal image (simulated survivor)
        cv2.circle(self.synth_thermal, (320, 256), 25, 255, -1)

    def test_01_cross_spectral_homography(self):
        """Validates that homography matrix is non-singular and warps thermal into RGB space."""
        registrar = CrossSpectralRegistrar(
            rgb_shape=(self.rgb_h, self.rgb_w),
            thermal_shape=(self.th_h, self.th_w),
        )

        # 1. Compute homography at 40m altitude
        H_40m = registrar.compute_homography(altitude_meters=40.0)
        self.assertEqual(H_40m.shape, (3, 3))
        self.assertAlmostEqual(H_40m[2, 2], 1.0, places=5)
        self.assertNotEqual(np.linalg.det(H_40m), 0.0)

        # 2. Warp thermal image
        warped = registrar.register_thermal_to_rgb(self.synth_thermal, altitude_meters=40.0)
        self.assertEqual(warped.shape, (self.rgb_h, self.rgb_w))
        # Valid hotspot must exist in warped image
        self.assertGreater(np.max(warped), 200)

        # 3. Spectral fusion
        fused = registrar.fuse_spectral_views(self.synth_rgb, self.synth_thermal, altitude_meters=40.0)
        self.assertEqual(fused.shape, (self.rgb_h, self.rgb_w, 3))
        self.assertEqual(fused.dtype, np.uint8)

    def test_02_adaptive_smoke_filter(self):
        """Validates visibility entropy and contrast enhancement."""
        smoke_filter = AdaptiveSmokeFilter()

        # 1. Clear image test
        entropy_clear, grad_clear, is_smoke = smoke_filter.calculate_scene_visibility(self.synth_rgb)
        self.assertGreater(entropy_clear, 0.0)
        self.assertFalse(is_smoke)

        # 2. Blurry uniform smoke frame
        hazy_frame = np.full((self.rgb_h, self.rgb_w, 3), 128, dtype=np.uint8)
        entropy_smoke, grad_smoke, is_smoke_flag = smoke_filter.calculate_scene_visibility(hazy_frame)
        self.assertTrue(is_smoke_flag)

        # 3. Enhancement routines
        enhanced_rgb = smoke_filter.enhance_optical_smoke(self.synth_rgb)
        self.assertEqual(enhanced_rgb.shape, self.synth_rgb.shape)

        enhanced_thermal = smoke_filter.enhance_thermal_contrast(self.synth_thermal)
        self.assertEqual(enhanced_thermal.shape, self.synth_thermal.shape)

    def test_03_mavlink_flight_dispatcher(self):
        """Validates flight command dispatching, loiter mode, and telemetry packets."""
        autopilot = MAVLinkAutopilotController(connection_string=None)
        telem = autopilot.get_telemetry()

        self.assertIsInstance(telem, TelemetryPacket)
        self.assertEqual(telem.flight_mode, "AUTO_MISSION")

        # Command loiter
        success = autopilot.command_loiter(target_gps=(34.05, -118.25))
        self.assertTrue(success)
        self.assertEqual(autopilot.current_telemetry.flight_mode, "LOITER_HOVER")

        # Command gimbal pitch/yaw
        success_gimbal = autopilot.command_gimbal_track(pitch_deg=-35.0, yaw_deg=12.0)
        self.assertTrue(success_gimbal)
        self.assertEqual(autopilot.current_telemetry.gimbal_pitch_deg, -35.0)

        # Resume mission
        autopilot.resume_mission()
        self.assertEqual(autopilot.current_telemetry.flight_mode, "AUTO_MISSION")
        self.assertGreaterEqual(len(autopilot.command_audit_log), 3)

    def test_04_fire_propagation_model(self):
        """Validates GSD scaling, hazard optical flow velocity, and arrival time estimation."""
        hazard_model = FirePropagationModel(focal_length_px=1450.0)

        # 1. GSD check
        gsd_45m = hazard_model.compute_ground_sampling_distance(altitude_m=45.0)
        self.assertAlmostEqual(gsd_45m, 45.0 / 1450.0, places=4)

        # 2. Simulated fire mask (active flame line)
        fire_mask = np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint8)
        cv2.rectangle(fire_mask, (100, 100), (300, 800), 255, -1)

        # Optical flow motion simulation across 2 frames
        frame1 = np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint8)
        frame2 = np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint8)
        cv2.rectangle(frame1, (100, 100), (300, 800), 220, -1)
        cv2.rectangle(frame2, (110, 100), (310, 800), 220, -1)  # 10px shift right

        metrics_frame1 = hazard_model.track_hazard_velocity(frame1, fire_mask, altitude_m=45.0)
        metrics_frame2 = hazard_model.track_hazard_velocity(frame2, fire_mask, altitude_m=45.0)

        self.assertGreater(metrics_frame2["velocity_mps"], 0.0)

        # 3. Time-to-impact calculation
        survivor_center = (960, 540)
        dist_m, time_min, urgency = hazard_model.estimate_time_to_impact(
            survivor_center_px=survivor_center,
            fire_mask=fire_mask,
            fire_velocity_mps=1.5,
            altitude_m=45.0
        )
        self.assertGreater(dist_m, 0.0)
        self.assertGreater(time_min, 0.0)
        self.assertIn(urgency, ["LOW", "MODERATE", "CRITICAL_IMMINENT"])

    def test_05_agentic_orchestrator_loop(self):
        """Validates the full perception-decision-action loop and human supervisor gateway."""
        orchestrator = AeroGuardianOrchestrator()
        self.assertEqual(orchestrator.current_state, MissionState.PATROL)

        # Cycle 1: Low-confidence detection triggers ACTIVE_INSPECTION & LOITER
        raw_detections = [{"class": "human", "confidence": 0.52, "bbox": [940, 520, 40, 80]}]
        fire_mask = np.zeros((self.rgb_h, self.rgb_w), dtype=np.uint8)
        cv2.rectangle(fire_mask, (200, 200), (400, 800), 255, -1)

        result_c1 = orchestrator.process_cycle(
            self.synth_rgb, self.synth_thermal, raw_detections, fire_mask
        )
        self.assertEqual(result_c1["current_state"], MissionState.ACTIVE_INSPECTION.value)
        self.assertIn("MAV_CMD_NAV_LOITER_UNLIM", result_c1["actions_dispatched"])

        # Cycle 2: Active Re-inspection confirms survivor, transitions to HAZARD_TRIAGE and HUMAN_GATE_PENDING
        result_c2 = orchestrator.process_cycle(
            self.synth_rgb, self.synth_thermal, raw_detections, fire_mask
        )
        self.assertEqual(result_c2["current_state"], MissionState.HUMAN_GATE_PENDING.value)
        self.assertIsNotNone(orchestrator.active_incident)

        # Cycle 3: Human supervisor approves rescue dispatch
        approval = orchestrator.approve_dispatch(commander_name="Chief Miller")
        self.assertEqual(approval["status"], "SUCCESS")
        self.assertEqual(orchestrator.current_state, MissionState.PATROL)

    def test_06_mavlink_gcs_bridge_udp_14550(self):
        """Validates MAVLink UDP Port 14550 Ground Control Station socket bridge."""
        from pymavlink import mavutil
        import socket

        # Find available ephemeral test port to avoid collision with running GCS
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(('127.0.0.1', 0))
            test_port = s.getsockname()[1]

        receiver = mavutil.mavlink_connection(f"udpin:127.0.0.1:{test_port}")
        autopilot = MAVLinkAutopilotController(
            gcs_target_ip="127.0.0.1",
            gcs_target_port=test_port,
            enable_gcs_bridge=True
        )

        try:
            # 1. Send HEARTBEAT
            sent_hb = autopilot.send_gcs_heartbeat()
            self.assertTrue(sent_hb)
            msg_hb = receiver.recv_match(type="HEARTBEAT", blocking=True, timeout=2.0)
            self.assertIsNotNone(msg_hb)
            self.assertEqual(msg_hb.get_type(), "HEARTBEAT")

            # 2. Send GLOBAL_POSITION_INT
            sent_pos = autopilot.send_gcs_global_position()
            self.assertTrue(sent_pos)
            msg_pos = receiver.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2.0)
            self.assertIsNotNone(msg_pos)
            self.assertEqual(msg_pos.get_type(), "GLOBAL_POSITION_INT")
            self.assertAlmostEqual(msg_pos.lat / 1e7, autopilot.current_telemetry.latitude, places=4)
            self.assertAlmostEqual(msg_pos.lon / 1e7, autopilot.current_telemetry.longitude, places=4)

            # 3. Test background bridge thread start and stop
            autopilot.start_gcs_bridge_thread(rate_hz=10.0)
            self.assertTrue(autopilot._bridge_running)
            time.sleep(0.15)
            autopilot.stop_gcs_bridge()
            self.assertFalse(autopilot._bridge_running)
        finally:
            autopilot.stop_gcs_bridge()
            receiver.close()

    def test_07_flame_wildfire_optical_flow_verification(self):
        """Validates FLAME wildfire dataset ingestion, hazard mask extraction, and dense optical flow."""
        from src.perception.dnn_detector import AeroTargetDetector

        # 1. Load FLAME aerial drone frame pair
        rgb = cv2.imread("data/samples/visible/wildfire.jpg")
        th = cv2.imread("data/samples/thermal/wildfire.jpg", cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(rgb, "Optical FLAME wildfire frame must load from disk")
        self.assertIsNotNone(th, "Thermal FLAME wildfire frame must load from disk")

        # 2. Extract hazard mask on authentic combustion thermal signatures (>220)
        hazard_model = FirePropagationModel(focal_length_px=1450.0)
        fire_mask = hazard_model.extract_hazard_mask(th, threshold=220)
        active_px = int(np.count_nonzero(fire_mask))
        self.assertGreaterEqual(active_px, 5000, "Active wildfire flame front must have >= 5000 combustion pixels")

        # 3. Farneback optical flow on flame front displacement
        flow_res1 = hazard_model.track_hazard_velocity(th, fire_mask, altitude_m=45.0, fps=30.0)
        self.assertEqual(flow_res1["velocity_mps"], 0.0)

        # Shift flame perimeter by 1.2 px east, -0.6 px north (wind-driven progression)
        M = np.float32([[1, 0, 1.2], [0, 1, -0.6]])
        th_shifted = cv2.warpAffine(th, M, (th.shape[1], th.shape[0]))
        mask_shifted = hazard_model.extract_hazard_mask(th_shifted, threshold=220)

        flow_res2 = hazard_model.track_hazard_velocity(th_shifted, mask_shifted, altitude_m=45.0, fps=30.0)
        self.assertGreater(flow_res2["velocity_mps"], 0.5)
        self.assertGreaterEqual(flow_res2["heading_deg"], 10.0)
        self.assertLessEqual(flow_res2["heading_deg"], 45.0)

        # 4. Multi-spectral target detector: confirms zero false human survivors inside raging forest fire
        detector = AeroTargetDetector()
        dets = detector.detect_multispectral(rgb, th)
        self.assertEqual(len(dets), 0, "No false humans or vehicles should be detected inside uninhabited forest burn")

        # 5. Tactical HUD banner verification
        registrar = CrossSpectralRegistrar()
        fused = registrar.fuse_spectral_views(rgb, th, altitude_meters=45.0)
        hud = registrar.render_tactical_hud(
            fused,
            telemetry={"altitude_m": 45.0},
            detections=dets,
            fire_metrics=flow_res2
        )
        self.assertGreaterEqual(hud.shape[1], 960)
        self.assertGreaterEqual(hud.shape[0], 540)

    def test_08_bidirectional_gcs_and_target_geolocation(self):
        """Validates target GPS projection, Lucas-Kanade ego-motion compensation, and bidirectional MAVLink commands."""
        hazard_model = FirePropagationModel(focal_length_px=1450.0)

        # 1. Target Geolocation photogrammetry check
        t_lat, t_lon = hazard_model.compute_target_gps(
            drone_lat=34.052235,
            drone_lon=-118.243683,
            altitude_m=45.0,
            target_pixel_xy=(261, 165),
            image_shape=(540, 960)
        )
        self.assertAlmostEqual(t_lat, 34.052264, places=5)
        self.assertAlmostEqual(t_lon, -118.243757, places=5)

        # 2. Ego-motion background feature compensation
        np.random.seed(42)
        bg = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
        for i in range(20, 280, 30):
            for j in range(20, 280, 30):
                cv2.circle(bg, (i, j), 4, 255, -1)

        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (180, 180), (280, 280), 255, -1)

        M = np.float32([[1, 0, 2.0], [0, 1, -1.0]])
        bg_shifted = cv2.warpAffine(bg, M, (300, 300))

        dx, dy, comp = hazard_model.estimate_drone_egomotion(bg, bg_shifted, mask)
        self.assertTrue(comp, "Ego-motion compensation must succeed on textured ground")
        self.assertAlmostEqual(dx, 2.0, delta=0.5)
        self.assertAlmostEqual(dy, -1.0, delta=0.5)

        # 2b. Heavy smoke condition: ground is obscured (<4 corners found) -> falls back to MAVLink groundspeed
        smoke_obscured = np.full((300, 300), 120, dtype=np.uint8)  # zero features
        dx_smoke, dy_smoke, comp_smoke = hazard_model.estimate_drone_egomotion(
            smoke_obscured, smoke_obscured, mask, mavlink_groundspeed_px=(1.75, -0.85)
        )
        self.assertTrue(comp_smoke, "Heavy smoke must trigger MAVLink fallback compensation")
        self.assertEqual(dx_smoke, 1.75)
        self.assertEqual(dy_smoke, -0.85)

        # 3. Bidirectional MAVLink command execution
        from pymavlink import mavutil
        import socket
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_sock.bind(('127.0.0.1', 0))
        test_port = test_sock.getsockname()[1]
        test_sock.close()

        autopilot = MAVLinkAutopilotController(
            gcs_target_ip="127.0.0.1",
            gcs_target_port=test_port,
            enable_gcs_bridge=True
        )

        mock_gcs = mavutil.mavlink_connection(f"udpin:127.0.0.1:{test_port}", source_system=255)

        # Autopilot broadcasts telemetry to establish connection
        autopilot.broadcast_gcs_telemetry(force=True)
        time.sleep(0.05)
        _ = mock_gcs.recv_match(blocking=False)

        # GCS sends Return-To-Launch command (MAV_CMD_NAV_RETURN_TO_LAUNCH = 20)
        mock_gcs.mav.command_long_send(1, 1, 20, 0, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(0.05)

        processed = autopilot.poll_incoming_gcs_commands()
        self.assertGreaterEqual(len(processed), 1)
        self.assertEqual(processed[0]["command"], "GCS_COMMAND_RTL")
        self.assertEqual(autopilot.current_telemetry.flight_mode, "RETURN_TO_LAUNCH")

        autopilot.stop_gcs_bridge()
        mock_gcs.close()


if __name__ == "__main__":
    unittest.main()



