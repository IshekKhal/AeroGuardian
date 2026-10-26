"""
AeroGuardian MAVLink Autopilot & Gimbal Actuation Controller
Communicates with ArduPilot / PX4 drone flight controllers over MAVLink (UDP/Serial).
Dispatches active flight commands (Loiter, Gimbal Mount Control, Digital Zoom)
and streams high-frequency vehicle telemetry.
"""

import os
os.environ["MAVLINK20"] = "1"

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import math
import threading
import time


@dataclass
class TelemetryPacket:
    latitude: float = 35.031500
    longitude: float = -111.597800
    altitude_m: float = 45.0
    heading_deg: float = 180.0
    gimbal_pitch_deg: float = -45.0
    gimbal_yaw_deg: float = 0.0
    battery_percent: float = 88.0
    flight_mode: str = "AUTO_MISSION"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "altitude_m": round(self.altitude_m, 1),
            "heading_deg": round(self.heading_deg, 1),
            "gimbal_pitch_deg": round(self.gimbal_pitch_deg, 1),
            "gimbal_yaw_deg": round(self.gimbal_yaw_deg, 1),
            "battery_percent": round(self.battery_percent, 1),
            "flight_mode": self.flight_mode,
            "timestamp": self.timestamp,
        }


class MAVLinkAutopilotController:
    """
    Autopilot and Gimbal command dispatcher.
    Supports live MAVLink UDP connection, standalone telemetry replay simulation,
    and a non-blocking Ground Control Station (GCS) UDP 14550 bridge for QGroundControl.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        gcs_target_ip: str = "127.0.0.1",
        gcs_target_port: int = 14550,
        enable_gcs_bridge: bool = True,
    ):
        """
        Args:
            connection_string: e.g. 'udpin:127.0.0.1:14551' or None for internal SITL simulation.
            gcs_target_ip: Destination IP for Ground Control Station (default '127.0.0.1').
            gcs_target_port: Standard MAVLink GCS UDP port (default 14550).
            enable_gcs_bridge: Whether to transmit live telemetry to GCS port 14550.
        """
        self.connection_string = connection_string
        self.gcs_target_ip = gcs_target_ip
        self.gcs_target_port = gcs_target_port
        self.enable_gcs_bridge = enable_gcs_bridge

        self.is_connected = False
        self.master = None
        self.gcs_conn = None
        self.current_telemetry = TelemetryPacket()
        self.command_audit_log: List[Dict[str, Any]] = []
        self._command_listeners: List[Any] = []

        self._last_heartbeat_time = 0.0
        self._last_position_time = 0.0
        self._bridge_running = False
        self._bridge_thread: Optional[threading.Thread] = None

        if self.connection_string:
            try:
                from pymavlink import mavutil
                self.master = mavutil.mavlink_connection(self.connection_string)
                self.is_connected = True
            except Exception:
                # Fallback to local SITL replay mode if live port is unavailable
                self.is_connected = False

        if self.enable_gcs_bridge:
            self._init_gcs_bridge()

    def _init_gcs_bridge(self):
        """Initializes non-blocking UDP broadcast connection to Ground Control Station."""
        try:
            from pymavlink import mavutil
            target = f"udpout:{self.gcs_target_ip}:{self.gcs_target_port}"
            self.gcs_conn = mavutil.mavlink_connection(target, source_system=1, source_component=1)
        except Exception:
            self.gcs_conn = None

    def send_gcs_heartbeat(self) -> bool:
        """
        Transmits non-blocking MAVLink HEARTBEAT packet to GCS (127.0.0.1:14550).
        Notifies QGroundControl / Mission Planner that vehicle is online and armed.
        """
        if not self.gcs_conn:
            return False
        try:
            from pymavlink import mavutil
            self.gcs_conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED | mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                3,  # AUTO / GUIDED flight mode
                mavutil.mavlink.MAV_STATE_ACTIVE
            )
            self._last_heartbeat_time = time.time()
            return True
        except Exception:
            return False

    def send_gcs_global_position(self) -> bool:
        """
        Transmits non-blocking MAVLink GLOBAL_POSITION_INT packet to GCS (127.0.0.1:14550).
        Supplies live latitude, longitude, altitude, and heading to GCS map display.
        """
        if not self.gcs_conn:
            return False
        try:
            telem = self.current_telemetry
            boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
            lat_int = int(telem.latitude * 1e7)
            lon_int = int(telem.longitude * 1e7)
            alt_mm = int(telem.altitude_m * 1000)
            hdg_cdeg = int(telem.heading_deg * 100)

            self.gcs_conn.mav.global_position_int_send(
                boot_ms,
                lat_int,
                lon_int,
                alt_mm,
                alt_mm,
                0, 0, 0,
                hdg_cdeg
            )
            self._last_position_time = time.time()
            return True
        except Exception:
            return False

    # Standard MAVLink parameters for Ground Control Stations (QGroundControl / Mission Planner)
    STANDARD_PARAMS = [
        (b"SYSID_THISMAV", 1.0, 6),       # MAV_PARAM_TYPE_INT32
        (b"SYSID_MYGCS", 255.0, 6),       # MAV_PARAM_TYPE_INT32
        (b"FRAME_CLASS", 1.0, 1),         # MAV_PARAM_TYPE_UINT8 (Quad)
        (b"FRAME_TYPE", 1.0, 1),          # MAV_PARAM_TYPE_UINT8 (X)
        (b"ARMING_CHECK", 0.0, 6),        # MAV_PARAM_TYPE_INT32 (Disabled/Ready)
        (b"BATT_CAPACITY", 5200.0, 9),    # MAV_PARAM_TYPE_REAL32 (mAh)
        (b"PILOT_SPEED_UP", 250.0, 9),    # MAV_PARAM_TYPE_REAL32 (cm/s)
        (b"WP_YAW_BEHAVIOR", 1.0, 1),     # MAV_PARAM_TYPE_UINT8
        (b"RTL_ALT", 4500.0, 9),          # MAV_PARAM_TYPE_REAL32 (cm)
        (b"FLTMODE1", 3.0, 1),            # AUTO
        (b"FLTMODE2", 5.0, 1),            # LOITER
        (b"FLTMODE3", 6.0, 1),            # RTL
    ]

    def send_gcs_vfr_hud(self) -> bool:
        """
        Transmits non-blocking VFR_HUD packet to GCS.
        Supplies airspeed, groundspeed, heading, throttle, and altitude.
        """
        if not self.gcs_conn:
            return False
        try:
            telem = self.current_telemetry
            spd = 12.5 if "AUTO" in telem.flight_mode else 0.0
            self.gcs_conn.mav.vfr_hud_send(
                spd,
                spd,
                int(telem.heading_deg),
                45,
                float(telem.altitude_m),
                0.0
            )
            return True
        except Exception:
            return False

    def send_gcs_attitude(self) -> bool:
        """
        Transmits non-blocking ATTITUDE packet to GCS.
        Supplies vehicle pitch, roll, and yaw angles.
        """
        if not self.gcs_conn:
            return False
        try:
            boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
            self.gcs_conn.mav.attitude_send(
                boot_ms,
                0.0,
                0.0,
                math.radians(self.current_telemetry.heading_deg),
                0.0,
                0.0,
                0.0
            )
            return True
        except Exception:
            return False

    def send_gcs_sys_status(self) -> bool:
        """
        Transmits non-blocking SYS_STATUS packet to GCS.
        Supplies battery percentage, voltage, and sensor readiness flags.
        """
        if not self.gcs_conn:
            return False
        try:
            sensors = 0x2F
            self.gcs_conn.mav.sys_status_send(
                sensors, sensors, sensors,
                320,
                22200,
                1250,
                int(self.current_telemetry.battery_percent),
                0, 0, 0, 0, 0, 0
            )
            return True
        except Exception:
            return False

    def broadcast_gcs_telemetry(self, force: bool = False) -> bool:
        """
        Sends non-blocking HEARTBEAT (~1 Hz), GLOBAL_POSITION_INT (~5 Hz),
        VFR_HUD, ATTITUDE, and SYS_STATUS to GCS UDP port 14550.
        """
        if not self.enable_gcs_bridge or not self.gcs_conn:
            return False
        now = time.time()
        sent = False
        if force or (now - self._last_heartbeat_time >= 1.0):
            if self.send_gcs_heartbeat():
                sent = True
            self.send_gcs_sys_status()
        if force or (now - self._last_position_time >= 0.20):
            if self.send_gcs_global_position():
                sent = True
            self.send_gcs_attitude()
            self.send_gcs_vfr_hud()
        return sent

    def start_gcs_bridge_thread(self, rate_hz: float = 5.0) -> None:
        """
        Spawns a daemon thread continuously transmitting telemetry to QGroundControl / Mission Planner
        and polling for incoming ground station commands.
        """
        if self._bridge_running:
            return
        self._bridge_running = True
        interval = 1.0 / max(rate_hz, 1.0)

        def _bridge_worker():
            while self._bridge_running:
                self.broadcast_gcs_telemetry(force=False)
                self.poll_incoming_gcs_commands()
                time.sleep(interval)

        self._bridge_thread = threading.Thread(
            target=_bridge_worker,
            daemon=True,
            name="MAVLinkGCSBridge"
        )
        self._bridge_thread.start()

    def stop_gcs_bridge(self) -> None:
        """Stops GCS background broadcast thread and closes network sockets."""
        self._bridge_running = False
        if self._bridge_thread and self._bridge_thread.is_alive():
            self._bridge_thread.join(timeout=0.5)
            self._bridge_thread = None
        if self.gcs_conn:
            try:
                self.gcs_conn.close()
            except Exception:
                pass
            self.gcs_conn = None

    def register_command_listener(self, listener) -> None:
        """Registers a callback function fn(command_name, params) for incoming GCS commands."""
        self._command_listeners.append(listener)

    def poll_incoming_gcs_commands(self) -> List[Dict[str, Any]]:
        """
        Polls non-blocking for incoming MAVLink commands from Ground Control Station.
        Handles mission protocol, parameters, autopilot capabilities, and flight commands.
        """
        if not self.gcs_conn:
            return []

        processed = []
        try:
            from pymavlink import mavutil
            while True:
                msg = self.gcs_conn.recv_msg()
                if msg is None:
                    break

                msg_type = msg.get_type()

                if msg_type == 'COMMAND_LONG':
                    cmd_id = msg.command
                    action_name = f"COMMAND_{cmd_id}"
                    params = {
                        "param1": msg.param1, "param2": msg.param2,
                        "param3": msg.param3, "param4": msg.param4,
                        "param5": msg.param5, "param6": msg.param6,
                        "param7": msg.param7
                    }

                    if cmd_id == 17:  # MAV_CMD_NAV_LOITER_UNLIM
                        action_name = "GCS_COMMAND_LOITER"
                        self.current_telemetry.flight_mode = "LOITER_HOVER"
                    elif cmd_id == 20:  # MAV_CMD_NAV_RETURN_TO_LAUNCH
                        action_name = "GCS_COMMAND_RTL"
                        self.current_telemetry.flight_mode = "RETURN_TO_LAUNCH"
                    elif cmd_id == 21:  # MAV_CMD_NAV_LAND
                        action_name = "GCS_COMMAND_LAND"
                        self.current_telemetry.flight_mode = "LAND"
                    elif cmd_id == 22:  # MAV_CMD_NAV_TAKEOFF
                        action_name = "GCS_COMMAND_TAKEOFF"
                        self.current_telemetry.flight_mode = "TAKEOFF"
                    elif cmd_id == 300:  # MAV_CMD_MISSION_START
                        action_name = "GCS_COMMAND_RESUME"
                        self.current_telemetry.flight_mode = "AUTO_MISSION"
                    elif cmd_id == 400:  # MAV_CMD_COMPONENT_ARM_DISARM
                        action_name = "GCS_COMMAND_ARM_DISPATCH" if msg.param1 == 1.0 else "GCS_COMMAND_DISARM"
                    elif cmd_id in (512, 520):  # AUTOPILOT_CAPABILITIES / REQUEST_MESSAGE
                        action_name = "GCS_REQUEST_CAPABILITIES"

                    # Acknowledge command back to GCS
                    try:
                        self.gcs_conn.mav.command_ack_send(
                            cmd_id,
                            mavutil.mavlink.MAV_RESULT_ACCEPTED
                        )
                    except Exception:
                        pass

                    self._record_command(action_name, params)
                    processed.append({"command": action_name, "params": params})

                    for listener in self._command_listeners:
                        try:
                            listener(action_name, params)
                        except Exception:
                            pass

                elif msg_type == 'SET_MODE':
                    self.current_telemetry.flight_mode = "GCS_CUSTOM_MODE"
                    self._record_command("GCS_SET_MODE", {"custom_mode": msg.custom_mode})
                    try:
                        self.gcs_conn.mav.command_ack_send(
                            176,  # MAV_CMD_DO_SET_MODE
                            mavutil.mavlink.MAV_RESULT_ACCEPTED
                        )
                    except Exception:
                        pass
                    processed.append({"command": "GCS_SET_MODE", "custom_mode": msg.custom_mode})

                elif msg_type == 'MISSION_REQUEST_LIST':
                    # QGroundControl requesting mission items, fence points, or rally points
                    m_type = getattr(msg, 'mission_type', 0)
                    try:
                        self.gcs_conn.mav.mission_count_send(
                            msg.get_srcSystem(),
                            msg.get_srcComponent(),
                            0,
                            mission_type=m_type
                        )
                    except Exception:
                        pass
                    self._record_command("GCS_MISSION_REQUEST_LIST", {"mission_type": m_type, "count": 0})
                    processed.append({"command": "GCS_MISSION_REQUEST_LIST", "mission_type": m_type})

                elif msg_type in ('MISSION_REQUEST', 'MISSION_REQUEST_INT'):
                    m_type = getattr(msg, 'mission_type', 0)
                    try:
                        self.gcs_conn.mav.mission_ack_send(
                            msg.get_srcSystem(),
                            msg.get_srcComponent(),
                            mavutil.mavlink.MAV_MISSION_ACCEPTED,
                            mission_type=m_type
                        )
                    except Exception:
                        pass

                elif msg_type == 'MISSION_CLEAR_ALL':
                    m_type = getattr(msg, 'mission_type', 0)
                    try:
                        self.gcs_conn.mav.mission_ack_send(
                            msg.get_srcSystem(),
                            msg.get_srcComponent(),
                            mavutil.mavlink.MAV_MISSION_ACCEPTED,
                            mission_type=m_type
                        )
                    except Exception:
                        pass
                    self._record_command("GCS_MISSION_CLEAR_ALL", {"mission_type": m_type})

                elif msg_type == 'PARAM_REQUEST_LIST':
                    # Transmit standard parameter table
                    total = len(self.STANDARD_PARAMS)
                    for idx, (p_id, p_val, p_type) in enumerate(self.STANDARD_PARAMS):
                        try:
                            self.gcs_conn.mav.param_value_send(
                                p_id, p_val, p_type, total, idx
                            )
                        except Exception:
                            pass
                    self._record_command("GCS_PARAM_REQUEST_LIST", {"params_sent": total})
                    processed.append({"command": "GCS_PARAM_REQUEST_LIST", "count": total})

                elif msg_type == 'PARAM_REQUEST_READ':
                    p_idx = getattr(msg, 'param_index', -1)
                    p_id_raw = getattr(msg, 'param_id', b'')
                    if isinstance(p_id_raw, str):
                        p_id_raw = p_id_raw.encode('ascii')
                    p_id_clean = p_id_raw.rstrip(b'\x00')

                    matched = None
                    total = len(self.STANDARD_PARAMS)
                    for idx, (p_id, p_val, p_type) in enumerate(self.STANDARD_PARAMS):
                        if idx == p_idx or p_id == p_id_clean:
                            matched = (p_id, p_val, p_type, idx)
                            break
                    if matched:
                        try:
                            self.gcs_conn.mav.param_value_send(
                                matched[0], matched[1], matched[2], total, matched[3]
                            )
                        except Exception:
                            pass

                elif msg_type == 'PING':
                    try:
                        self.gcs_conn.mav.ping_send(
                            getattr(msg, 'time_usec', 0),
                            getattr(msg, 'seq', 0),
                            msg.get_srcSystem(),
                            msg.get_srcComponent(),
                        )
                    except Exception:
                        pass

        except Exception:
            pass

        return processed

    def get_telemetry(self) -> TelemetryPacket:
        """
        Fetches current telemetry packet from drone or updates simulated flight path.
        Automatically broadcasts updated coordinates to GCS bridge.
        """
        if self.is_connected and self.master:
            try:
                msg = self.master.recv_match(type=['GLOBAL_POSITION_INT', 'ATTITUDE', 'SYS_STATUS'], blocking=False)
                if msg:
                    if msg.get_type() == 'GLOBAL_POSITION_INT':
                        self.current_telemetry.latitude = msg.lat / 1e7
                        self.current_telemetry.longitude = msg.lon / 1e7
                        self.current_telemetry.altitude_m = msg.relative_alt / 1e3
                        self.current_telemetry.heading_deg = msg.hdg / 100.0
                    elif msg.get_type() == 'SYS_STATUS':
                        self.current_telemetry.battery_percent = float(msg.battery_remaining)
            except Exception:
                pass
        self.current_telemetry.timestamp = time.time()
        if not self._bridge_running:
            self.broadcast_gcs_telemetry(force=False)
        return self.current_telemetry

    def command_loiter(self, target_gps: Optional[Tuple[float, float]] = None) -> bool:
        """
        Halts forward flight and commands drone to loiter / hover over current or target position.
        MAVLink Command: MAV_CMD_NAV_LOITER_UNLIM (#17)
        """
        lat = target_gps[0] if target_gps else self.current_telemetry.latitude
        lon = target_gps[1] if target_gps else self.current_telemetry.longitude

        self.current_telemetry.flight_mode = "LOITER_HOVER"
        self._record_command("NAV_LOITER_UNLIM", {"lat": lat, "lon": lon})
        self.broadcast_gcs_telemetry(force=True)

        if self.is_connected and self.master:
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    17,  # MAV_CMD_NAV_LOITER_UNLIM
                    0, 0, 0, 0, 0, lat, lon, self.current_telemetry.altitude_m
                )
                return True
            except Exception:
                return False
        return True

    def command_return_to_launch(self) -> bool:
        """
        Commands drone to abort current operation and Return To Launch (RTL).
        MAVLink Command: MAV_CMD_NAV_RETURN_TO_LAUNCH (#20)
        """
        self.current_telemetry.flight_mode = "RETURN_TO_LAUNCH"
        self._record_command("NAV_RETURN_TO_LAUNCH", {"action": "RTL"})
        self.broadcast_gcs_telemetry(force=True)

        if self.is_connected and self.master:
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    20,  # MAV_CMD_NAV_RETURN_TO_LAUNCH
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                return True
            except Exception:
                return False
        return True

    def command_gimbal_track(self, pitch_deg: float, yaw_deg: float) -> bool:
        """
        Points camera gimbal directly at target angular offset.
        MAVLink Command: MAV_CMD_DO_MOUNT_CONTROL (#205)
        Pitch: -90 (nadir/straight down) to 0 (horizontal).
        Yaw: -180 to +180 relative to body.
        """
        pitch_clamped = max(-90.0, min(0.0, pitch_deg))
        yaw_clamped = max(-180.0, min(180.0, yaw_deg))

        self.current_telemetry.gimbal_pitch_deg = pitch_clamped
        self.current_telemetry.gimbal_yaw_deg = yaw_clamped
        self._record_command("DO_MOUNT_CONTROL", {"pitch": pitch_clamped, "yaw": yaw_clamped})
        self.broadcast_gcs_telemetry(force=True)

        if self.is_connected and self.master:
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    205,  # MAV_CMD_DO_MOUNT_CONTROL
                    0,
                    pitch_clamped * 100,  # Centidegrees
                    0.0,
                    yaw_clamped * 100,
                    0, 0, 0, 2  # MAV_MOUNT_MODE_MAVLINK_TARGETING
                )
                return True
            except Exception:
                return False
        return True

    def resume_mission(self) -> bool:
        """
        Resumes pre-programmed autonomous search grid after target verification.
        MAVLink Command: MAV_CMD_MISSION_START (#300)
        """
        self.current_telemetry.flight_mode = "AUTO_MISSION"
        self._record_command("MISSION_START", {"status": "resumed"})
        self.broadcast_gcs_telemetry(force=True)

        if self.is_connected and self.master:
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    300,  # MAV_CMD_MISSION_START
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                return True
            except Exception:
                return False
        return True

    def _record_command(self, action: str, params: Dict[str, Any]):
        from datetime import timezone
        entry = {
            "action": action,
            "params": params,
            "timestamp": time.time(),
            "iso_time": datetime.now(timezone.utc).isoformat()
        }
        self.command_audit_log.append(entry)
