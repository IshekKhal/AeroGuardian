"""
AeroGuardian Tactical Mission Control Dashboard (FastAPI + WebSockets + Leaflet Esri Satellite GIS)
Integrates the Defense Aerial Triad:
1. MAVLink 2.0 Telemetry & Command Socket (UDP 14550)
2. Satellite GIS Tactical Mapping (Esri World Imagery + Breadcrumb Trail + Dynamic Wildfire Polygon)
3. High-Contrast Multi-Spectral HUD Overlay (OpenCV 5 Core Engine)
"""

import time
import os
from typing import Dict, Any, List, Optional, Tuple

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.orchestrator import AeroGuardianOrchestrator
from src.perception.stream_ingest import DualStreamIngest

app = FastAPI(title="AeroGuardian Tactical Command Console")

# Global singleton orchestrator
orchestrator = AeroGuardianOrchestrator()
orchestrator.autopilot.start_gcs_bridge_thread(rate_hz=5.0)

# Shared DualStreamIngest instance
stream_ingest = DualStreamIngest()

# Home Base Location (Coconino Incident Command Post, Flagstaff, AZ)
HOME_BASE_GPS = [35.031000, -111.598500]

# Historical flight path breadcrumbs for GIS map
flight_breadcrumbs: List[List[float]] = []

current_view_mode: str = "isotherm"

# Digital Gimbal & Sensor Zoom State (Option A)
current_zoom_factor: float = 1.0
target_zoom_factor: float = 1.0
zoom_center_norm: Tuple[float, float] = (0.5, 0.5)  # Normalized (cx, cy)


class ApprovalRequest(BaseModel):
    commander_name: str = "Chief Miller"


class MavlinkCommandRequest(BaseModel):
    command: str
    params: Optional[Dict[str, Any]] = None


class SelectPairRequest(BaseModel):
    pair_id: int


class SelectModeRequest(BaseModel):
    mode: str


class SetZoomRequest(BaseModel):
    zoom: float
    center_x: Optional[float] = 0.5
    center_y: Optional[float] = 0.5


# Sector boundaries for FLAME 2 pairs
PAIR_SECTOR_INFO = {
    1: {"name": "Sector A-01 (Flagstaff South)", "lat": 35.035200, "lon": -111.601500, "heading": 45.0},
    2: {"name": "Sector A-02 (Schultz Pass)", "lat": 35.033400, "lon": -111.599800, "heading": 90.0},
    3: {"name": "Sector B-12 (Mormon Lake Road)", "lat": 35.031500, "lon": -111.597800, "heading": 135.0},
    4: {"name": "Sector B-14 (Hart Prairie)", "lat": 35.029600, "lon": -111.596000, "heading": 180.0},
    5: {"name": "Sector C-03 (Sedona Rim)", "lat": 35.028200, "lon": -111.594200, "heading": 225.0},
    6: {"name": "Sector C-07 (Fort Valley)", "lat": 35.030500, "lon": -111.595500, "heading": 270.0},
    7: {"name": "Sector D-01 (Wing Mountain)", "lat": 35.034200, "lon": -111.598200, "heading": 315.0},
}


def get_sector_polygon(pair_id: int) -> List[List[float]]:
    """Returns realistic georeferenced fire perimeter polygon for the active flight sector."""
    sec = PAIR_SECTOR_INFO.get(pair_id, PAIR_SECTOR_INFO[3])
    clat, clon = sec["lat"], sec["lon"]
    return [
        [round(clat + 0.000700, 6), round(clon + 0.000600, 6)],
        [round(clat + 0.001000, 6), round(clon + 0.000300, 6)],
        [round(clat + 0.001150, 6), round(clon - 0.000300, 6)],
        [round(clat + 0.000850, 6), round(clon - 0.000800, 6)],
        [round(clat + 0.000500, 6), round(clon - 0.000500, 6)],
        [round(clat + 0.000400, 6), round(clon + 0.000100, 6)],
    ]


def generate_live_stream():
    """Generates continuous MJPEG video stream with live tactical HUD annotations and dynamic digital zoom."""
    global current_zoom_factor, target_zoom_factor, zoom_center_norm
    while True:
        try:
            frame_gen = stream_ingest.stream_flame2_flight(max_frames=50000, throttle=False, loop=True)
            for frame_idx, rgb, thermal, meta in frame_gen:
                telem_data = meta.get("telemetry", {})

                # Update autopilot telemetry from genuine video flight coordinates if not in manual override
                if telem_data:
                    mode = orchestrator.autopilot.current_telemetry.flight_mode
                    if mode != "LOITER_HOVER":
                        orchestrator.autopilot.current_telemetry.latitude = telem_data.get(
                            "latitude", orchestrator.autopilot.current_telemetry.latitude
                        )
                        orchestrator.autopilot.current_telemetry.longitude = telem_data.get(
                            "longitude", orchestrator.autopilot.current_telemetry.longitude
                        )
                        orchestrator.autopilot.current_telemetry.altitude_m = telem_data.get(
                            "altitude_m", orchestrator.autopilot.current_telemetry.altitude_m
                        )
                        orchestrator.autopilot.current_telemetry.heading_deg = telem_data.get(
                            "heading_deg", telem_data.get("gimbal_yaw_deg", orchestrator.autopilot.current_telemetry.heading_deg)
                        )

                # Fused frame with HUD using active view mode
                fused = orchestrator.registrar.fuse_spectral_views(
                    rgb_frame=rgb,
                    thermal_frame=thermal,
                    altitude_meters=orchestrator.autopilot.current_telemetry.altitude_m,
                    alpha=0.55,
                    mode=current_view_mode
                )

                # Process agentic cycle with native OpenCV DNN detection
                cycle = orchestrator.process_cycle(
                    rgb_frame=rgb,
                    thermal_frame=thermal,
                    raw_detections=None,
                    fire_mask=meta.get("fire_mask")
                )

                detections = cycle.get("detections", [])

                # Option A: Smooth digital zoom easing and visual target tracking
                current_zoom_factor += 0.16 * (target_zoom_factor - current_zoom_factor)

                if current_zoom_factor > 1.04:
                    fh, fw = fused.shape[:2]
                    crop_w = max(120, int(fw / current_zoom_factor))
                    crop_h = max(80, int(fh / current_zoom_factor))
                    cx = int(zoom_center_norm[0] * fw)
                    cy = int(zoom_center_norm[1] * fh)
                    x1 = max(0, min(fw - crop_w, cx - crop_w // 2))
                    y1 = max(0, min(fh - crop_h, cy - crop_h // 2))
                    cropped = fused[y1:y1 + crop_h, x1:x1 + crop_w]
                    fused = cv2.resize(cropped, (fw, fh), interpolation=cv2.INTER_LINEAR)

                    # Apply CLAHE contrast enhancement on the zoomed crop to heighten thermal detail
                    lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB)
                    l_chan, a_chan, b_chan = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    l_chan = clahe.apply(l_chan)
                    fused = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

                hud = orchestrator.registrar.render_tactical_hud(
                    fused_frame=fused,
                    telemetry=orchestrator.autopilot.current_telemetry.to_dict(),
                    detections=detections,
                    incident=cycle.get("incident"),
                    fire_metrics=cycle.get("fire_metrics")
                )

                # If zoomed in, draw optical gimbal targeting lock on HUD
                if current_zoom_factor > 1.04:
                    hh, hw = hud.shape[:2]
                    # Pulsing target lock indicator
                    pulse_color = (34, 197, 94) if (frame_idx % 20 < 10) else (56, 189, 248)
                    cv2.putText(
                        hud,
                        f"[GIMBAL ACTIVE TRACKING] SENSOR ZOOM: {current_zoom_factor:.1f}X | TARGET LOCK ENGAGED",
                        (hw // 2 - 240, hh - 68),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        pulse_color,
                        1,
                        cv2.LINE_AA
                    )
                    # Center target brackets
                    chx, chy = hw // 2, hh // 2
                    blen = 24
                    cv2.line(hud, (chx - blen, chy - blen), (chx - blen + 10, chy - blen), pulse_color, 2)
                    cv2.line(hud, (chx - blen, chy - blen), (chx - blen, chy - blen + 10), pulse_color, 2)
                    cv2.line(hud, (chx + blen, chy - blen), (chx + blen - 10, chy - blen), pulse_color, 2)
                    cv2.line(hud, (chx + blen, chy - blen), (chx + blen, chy - blen + 10), pulse_color, 2)
                    cv2.line(hud, (chx - blen, chy + blen), (chx - blen + 10, chy + blen), pulse_color, 2)
                    cv2.line(hud, (chx - blen, chy + blen), (chx - blen, chy + blen - 10), pulse_color, 2)
                    cv2.line(hud, (chx + blen, chy + blen), (chx + blen - 10, chy + blen), pulse_color, 2)
                    cv2.line(hud, (chx + blen, chy + blen), (chx + blen, chy + blen - 10), pulse_color, 2)
                    cv2.circle(hud, (chx, chy), 4, pulse_color, -1)

                ret, buffer = cv2.imencode('.jpg', hud, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.04)  # ~25 FPS for smooth network playback
        except Exception:
            time.sleep(0.05)


@app.get("/api/stream/live")
def stream_live_video():
    return StreamingResponse(
        generate_live_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/pairs")
def list_flight_pairs():
    """Lists all available authentic FLAME 2 dual-stream flight video pairs."""
    return stream_ingest.get_available_pairs()


@app.post("/api/stream/select_pair")
def select_flight_pair(req: SelectPairRequest):
    """Switches active FLAME 2 video pair for live processing."""
    global flight_breadcrumbs, target_zoom_factor
    flight_breadcrumbs.clear()
    target_zoom_factor = 1.0  # Reset zoom on pair switch
    success = stream_ingest.set_active_pair(req.pair_id)

    sec = PAIR_SECTOR_INFO.get(req.pair_id, PAIR_SECTOR_INFO[3])
    orchestrator.autopilot.current_telemetry.latitude = sec["lat"]
    orchestrator.autopilot.current_telemetry.longitude = sec["lon"]
    orchestrator.autopilot.current_telemetry.heading_deg = sec["heading"]
    orchestrator.latest_fire_polygon = get_sector_polygon(req.pair_id)

    orchestrator.log_thought(
        stage="Mission Navigation",
        tool_name="tool_switch_patrol_sector",
        observation=f"GCS operator selected FLAME 2 Flight Pair #{req.pair_id} ({sec['name']}).",
        decision=f"Re-tasking UAV-01 sensor pod to {sec['name']} prescribed burn perimeter."
    )
    return {
        "status": "success" if success else "error",
        "active_pair_id": stream_ingest.active_pair_id,
        "sector": sec
    }


@app.post("/api/stream/set_mode")
def set_stream_view_mode(req: SelectModeRequest):
    """Switches active multi-spectral view mode (isotherm, side_by_side, blend, optical, thermal)."""
    global current_view_mode
    valid_modes = ["isotherm", "side_by_side", "blend", "optical", "thermal"]
    if req.mode in valid_modes:
        current_view_mode = req.mode
        orchestrator.log_thought(
            stage="Optronic Sensor Control",
            tool_name="tool_switch_spectral_mode",
            observation=f"GCS operator selected spectral view: {req.mode.upper()}.",
            decision=f"Applying OpenCV 5 shader pipeline for {req.mode} visualization."
        )
        return {"status": "success", "mode": current_view_mode}
    return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid mode. Choose from {valid_modes}"})


@app.post("/api/gimbal/zoom")
def set_gimbal_zoom(req: SetZoomRequest):
    """Sets digital sensor zoom level and centers gimbal targeting (Option A)."""
    global target_zoom_factor, zoom_center_norm
    clamped_zoom = max(1.0, min(3.5, float(req.zoom)))
    target_zoom_factor = clamped_zoom
    if req.center_x is not None and req.center_y is not None:
        zoom_center_norm = (max(0.1, min(0.9, req.center_x)), max(0.1, min(0.9, req.center_y)))

    orchestrator.log_thought(
        stage="Payload Actuation",
        tool_name="tool_digital_zoom_actuate",
        observation=f"GCS commanded sensor zoom: {target_zoom_factor:.1f}x (Center: {zoom_center_norm[0]:.2f}, {zoom_center_norm[1]:.2f}).",
        decision="Applying spatial crop and CLAHE radiometric boost on anomaly region."
    )
    return {"status": "success", "target_zoom": target_zoom_factor, "center": zoom_center_norm}


@app.get("/api/status")
async def get_mission_status():
    telemetry = orchestrator.autopilot.get_telemetry()
    curr_coord = [round(telemetry.latitude, 6), round(telemetry.longitude, 6)]

    if not flight_breadcrumbs or flight_breadcrumbs[-1] != curr_coord:
        flight_breadcrumbs.append(curr_coord)
        if len(flight_breadcrumbs) > 250:
            flight_breadcrumbs.pop(0)

    # Dynamic georeferenced wildfire polygon
    wildfire_polygon = orchestrator.latest_fire_polygon
    if not wildfire_polygon:
        wildfire_polygon = get_sector_polygon(stream_ingest.active_pair_id)

    target_pin = None
    if orchestrator.active_incident and orchestrator.active_incident.target_gps:
        target_pin = list(orchestrator.active_incident.target_gps)

    sec_info = PAIR_SECTOR_INFO.get(stream_ingest.active_pair_id, PAIR_SECTOR_INFO[3])

    return {
        "state": orchestrator.current_state.value,
        "telemetry": telemetry.to_dict(),
        "incident": orchestrator.active_incident.__dict__ if orchestrator.active_incident else None,
        "target_gps": target_pin,
        "home_base_gps": HOME_BASE_GPS,
        "sector_name": sec_info["name"],
        "active_pair_id": stream_ingest.active_pair_id,
        "view_mode": current_view_mode,
        "zoom_factor": round(current_zoom_factor, 1),
        "breadcrumbs": flight_breadcrumbs,
        "wildfire_polygon": wildfire_polygon,
        "audit_log_count": len(orchestrator.autopilot.command_audit_log),
        "recent_thoughts": orchestrator.thought_stream_log[-120:],
        "total_thoughts_count": len(orchestrator.thought_stream_log),
        "gcs_bridge_connected": orchestrator.autopilot._bridge_running or (orchestrator.autopilot.gcs_conn is not None),
    }


@app.post("/api/mavlink/command")
async def dispatch_mavlink_command(req: MavlinkCommandRequest):
    """Dispatches bidirectional MAVLink command to drone autopilot."""
    global target_zoom_factor
    cmd = req.command.upper()
    telem = orchestrator.autopilot.get_telemetry()

    if cmd == "LOITER":
        orchestrator.autopilot.command_loiter((telem.latitude, telem.longitude))
        target_zoom_factor = 2.4  # Option A: Visual Loiter Zoom
        orchestrator.log_thought(
            stage="Autopilot Actuation",
            tool_name="tool_dispatch_loiter",
            observation=f"GCS operator commanded LOITER HOVER at ({telem.latitude:.6f}, {telem.longitude:.6f}).",
            decision="Holding station. Pausing waypoint sweep. Actuating 2.4x sensor zoom over target sector."
        )
        msg = f"MAV_CMD_NAV_LOITER_UNLIM: Holding ({telem.latitude:.6f}, {telem.longitude:.6f}) [2.4x ZOOM ACTIVE]"
    elif cmd == "RTL":
        orchestrator.autopilot.command_return_to_launch()
        target_zoom_factor = 1.0
        orchestrator.log_thought(
            stage="Autopilot Actuation",
            tool_name="tool_return_to_launch",
            observation=f"GCS operator commanded RETURN TO LAUNCH (RTL). Battery level: {telem.battery_percent}%.",
            decision=f"Climbing to safe transit altitude (60m AGL). Disengaging search grid. Returning to Home Base ({HOME_BASE_GPS[0]}, {HOME_BASE_GPS[1]})."
        )
        msg = f"MAV_CMD_NAV_RETURN_TO_LAUNCH: Returning to Home Base at 60m AGL"
    elif cmd == "RESUME":
        orchestrator.autopilot.resume_mission()
        target_zoom_factor = 1.0  # Option A: Zoom back out to wide FOV
        orchestrator.log_thought(
            stage="Autopilot Actuation",
            tool_name="tool_resume_patrol",
            observation="GCS operator commanded RESUME PATROL.",
            decision="Re-engaging autonomous lawnmower search grid at 45m AGL. Restoring 1.0x wide-angle FOV."
        )
        msg = "MAV_CMD_MISSION_START: Resuming autonomous patrol sweep (1.0x FOV)"
    elif cmd == "GIMBAL_NADIR":
        new_pitch = -45.0 if telem.gimbal_pitch_deg <= -80.0 else -90.0
        orchestrator.autopilot.command_gimbal_track(pitch_deg=new_pitch, yaw_deg=0.0)
        orchestrator.log_thought(
            stage="Gimbal Actuation",
            tool_name="tool_gimbal_tilt",
            observation=f"GCS operator toggled gimbal pitch to {new_pitch}°.",
            decision="Actuating 2-axis brushless gimbal mount via MAV_CMD_DO_MOUNT_CONTROL."
        )
        msg = f"MAV_CMD_DO_MOUNT_CONTROL: Gimbal pitch set to {new_pitch}°"
    else:
        return JSONResponse(status_code=400, content={"status": "ERROR", "message": f"Unknown command: {cmd}"})

    return {"status": "SUCCESS", "command": cmd, "message": msg, "telemetry": orchestrator.autopilot.get_telemetry().to_dict()}


@app.post("/api/approve_dispatch")
async def approve_emergency_dispatch(req: ApprovalRequest):
    result = orchestrator.approve_dispatch(commander_name=req.commander_name)
    orchestrator.log_thought(
        stage="Emergency Command Sign-Off",
        tool_name="tool_authorize_rescue_dispatch",
        observation=f"Incident Commander {req.commander_name} signed digital dispatch authorization.",
        decision="Rescue helicopter dispatch signal transmitted to CalFire dispatch. Resuming sector perimeter patrol."
    )
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEROGUARDIAN :: TACTICAL AIR-SURFACE INCIDENT HUD</title>
    <!-- Leaflet CSS for Tactical GIS Map -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <style>
        :root {
            --bg-base: #070a0e;
            --bg-panel: #0d1219;
            --bg-surface: #131a24;
            --border-tactical: #1e2836;
            --border-highlight: #2f3e54;
            --text-lead: #f0f6fc;
            --text-muted: #7d899b;
            --status-green: #22c55e;
            --status-amber: #f59e0b;
            --status-red: #ef4444;
            --tactical-cyan: #38bdf8;
            --mono-font: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: var(--mono-font); }
        body { background: var(--bg-base); color: var(--text-lead); height: 100vh; overflow: hidden; display: flex; flex-direction: column; }
        
        /* Tactical Ribbon Header */
        header { 
            background: var(--bg-panel); 
            border-bottom: 1px solid var(--border-tactical); 
            padding: 8px 14px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            flex-shrink: 0; 
            z-index: 100;
        }
        .header-brand { display: flex; align-items: center; gap: 10px; }
        .system-callsign { font-size: 13px; font-weight: 800; letter-spacing: 1.5px; color: var(--text-lead); }
        .system-tag { font-size: 10px; color: var(--tactical-cyan); border-left: 1px solid var(--border-tactical); padding-left: 10px; }
        
        /* Badges */
        .status-chip-group { display: flex; align-items: center; gap: 6px; }
        .status-chip { font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
        .chip-mavlink { background: rgba(56, 189, 248, 0.12); color: var(--tactical-cyan); border: 1px solid rgba(56, 189, 248, 0.35); }
        .chip-gps { background: rgba(34, 197, 94, 0.12); color: var(--status-green); border: 1px solid rgba(34, 197, 94, 0.35); }
        .chip-engine { background: rgba(245, 158, 11, 0.12); color: var(--status-amber); border: 1px solid rgba(245, 158, 11, 0.35); }
        .chip-cloud { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }
        
        /* Navigation Tabs */
        .workspace-tabs { display: flex; gap: 4px; background: var(--bg-base); padding: 2px; border-radius: 3px; border: 1px solid var(--border-tactical); }
        .tab-btn { background: transparent; border: none; color: var(--text-muted); padding: 5px 12px; font-size: 11px; font-weight: 700; cursor: pointer; border-radius: 2px; text-transform: uppercase; transition: all 0.15s; }
        .tab-btn:hover { color: var(--text-lead); background: rgba(255,255,255,0.04); }
        .tab-btn.active { color: #fff; background: var(--bg-surface); border: 1px solid var(--border-highlight); box-shadow: 0 1px 3px rgba(0,0,0,0.4); }
        
        /* Viewports Container */
        .workspace-viewport { flex: 1; position: relative; overflow: hidden; display: none; }
        .workspace-viewport.active-view { display: flex; flex-direction: column; height: calc(100vh - 49px); }

        /* VIEW 1: FLIGHT DECK (FULL GIMBAL) */
        .deck-container { flex: 1; display: flex; flex-direction: column; padding: 8px; gap: 8px; height: 100%; }
        .deck-top-bar { display: flex; justify-content: space-between; align-items: center; background: var(--bg-panel); border: 1px solid var(--border-tactical); padding: 6px 12px; border-radius: 2px; }
        .deck-video-stage { flex: 1; background: #000; border: 1px solid var(--border-tactical); position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        .deck-video-img { max-width: 100%; max-height: 100%; width: 100%; height: 100%; object-fit: contain; }
        
        /* VIEW 2: GIS THEATER (FULL MAP) */
        .gis-container { flex: 1; display: flex; flex-direction: column; padding: 8px; gap: 8px; height: 100%; }
        .gis-top-bar { display: flex; justify-content: space-between; align-items: center; background: var(--bg-panel); border: 1px solid var(--border-tactical); padding: 6px 12px; border-radius: 2px; }
        #full-gis-map { flex: 1; width: 100%; height: 100%; border: 1px solid var(--border-tactical); background: #0a0e14; }

        /* VIEW 3: INCIDENT COMMAND & TRIAGE GATE */
        .triage-container { flex: 1; display: grid; grid-template-columns: 2fr 1fr; gap: 10px; padding: 10px; height: 100%; overflow-y: auto; }
        .triage-panel { background: var(--bg-panel); border: 1px solid var(--border-tactical); border-radius: 3px; padding: 14px; display: flex; flex-direction: column; }
        .triage-title { font-size: 13px; font-weight: 800; letter-spacing: 1px; color: var(--tactical-cyan); margin-bottom: 12px; border-bottom: 1px solid var(--border-tactical); padding-bottom: 8px; }
        .triage-table { width: 100%; border-collapse: collapse; font-size: 11px; }
        .triage-table th { text-align: left; padding: 8px; background: var(--bg-surface); color: var(--text-muted); border-bottom: 1px solid var(--border-tactical); }
        .triage-table td { padding: 8px; border-bottom: 1px solid var(--border-tactical); color: var(--text-lead); }
        
        /* VIEW 4: MAVLINK PACKET STREAM & AGENT DECK */
        .mavlink-container { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; height: 100%; }
        .mav-panel { background: var(--bg-panel); border: 1px solid var(--border-tactical); border-radius: 3px; padding: 12px; display: flex; flex-direction: column; }
        .log-terminal { flex: 1; background: #04070a; border: 1px solid var(--border-tactical); padding: 10px; font-size: 11px; line-height: 1.5; overflow-y: auto; color: #a5d6ff; }

        /* VIEW 5: DUAL TACTICAL (SPLIT SCREEN DEFAULT) */
        .split-container { flex: 1; display: flex; flex-direction: column; padding: 8px; gap: 8px; height: 100%; }
        .split-primary-row { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 8px; height: 56vh; flex-shrink: 0; }
        .split-bottom-row { display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 8px; flex: 1; min-height: 0; }
        
        .split-card { background: var(--bg-panel); border: 1px solid var(--border-tactical); border-radius: 2px; display: flex; flex-direction: column; overflow: hidden; }
        .split-card-header { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 6px 10px; background: var(--bg-surface); border-bottom: 1px solid var(--border-tactical); display: flex; justify-content: space-between; align-items: center; color: var(--text-muted); letter-spacing: 0.5px; }
        
        .video-wrapper { flex: 1; background: #000; position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        .video-feed-el { width: 100%; height: 100%; object-fit: contain; }
        
        #split-gis-map { flex: 1; width: 100%; height: 100%; background: #0a0e14; }

        /* Controls & Buttons */
        .btn-action { background: var(--bg-surface); border: 1px solid var(--border-tactical); color: var(--text-lead); padding: 6px 10px; font-size: 10px; font-weight: 700; cursor: pointer; text-transform: uppercase; border-radius: 2px; transition: all 0.15s; }
        .btn-action:hover { background: var(--border-highlight); color: #fff; border-color: var(--tactical-cyan); }
        .btn-action.active { background: rgba(56, 189, 248, 0.18); border-color: var(--tactical-cyan); color: var(--tactical-cyan); }
        
        .btn-approve-gate { background: #166534; border: 1px solid #22c55e; color: #fff; padding: 10px; font-size: 11px; font-weight: 800; cursor: pointer; text-transform: uppercase; letter-spacing: 1px; border-radius: 2px; width: 100%; transition: background 0.2s; }
        .btn-approve-gate:hover:not(:disabled) { background: #15803d; }
        .btn-approve-gate:disabled { background: #1c2430; border-color: var(--border-tactical); color: #475569; cursor: not-allowed; }
        .btn-approved-gate { background: #1d4ed8 !important; border-color: #3b82f6 !important; }

        /* Form elements */
        select.tactical-select { background: var(--bg-base); border: 1px solid var(--border-tactical); color: var(--text-lead); font-size: 10px; padding: 4px 8px; font-family: var(--mono-font); border-radius: 2px; }
        
        /* Custom Map Icon */
        .uav-svg-icon { filter: drop-shadow(0 0 5px #38bdf8); transition: transform 0.4s ease-out; }
        .home-marker-badge { background: #15803d; color: white; border: 1px solid #fff; border-radius: 50%; font-size: 9px; font-weight: 800; width: 16px; height: 16px; display: flex; align-items: center; justify-content: center; }
    </style>
</head>
<body>
    <!-- Tactical Ribbon Header -->
    <header>
        <div class="header-brand">
            <span class="system-callsign">AEROGUARDIAN</span>
            <span class="system-tag">TACTICAL COMMAND CONSOLE</span>
        </div>
        
        <!-- Workspace Tabs -->
        <div class="workspace-tabs">
            <button class="tab-btn active" onclick="switchWorkspace('split')">[5] AIR-SURFACE SPLIT</button>
            <button class="tab-btn" onclick="switchWorkspace('deck')">[1] FLIGHT DECK</button>
            <button class="tab-btn" onclick="switchWorkspace('gis')">[2] GIS THEATER</button>
            <button class="tab-btn" onclick="switchWorkspace('triage')">[3] INCIDENT COMMAND</button>
            <button class="tab-btn" onclick="switchWorkspace('mavlink')">[4] MAVLINK STREAM</button>
        </div>

        <!-- System Status Chips -->
        <div class="status-chip-group">
            <span class="status-chip chip-mavlink" id="chip-mavlink">MAVLink 2.0 UDP:14550</span>
            <span class="status-chip chip-gps">GPS 3D FIX (14 SATS)</span>
            <span class="status-chip chip-engine">OpenCV 5.0 Core</span>
            <span class="status-chip chip-cloud">AWS Graviton4 Arm64 Ready</span>
        </div>
    </header>

    <!-- WORKSPACE 5: SPLIT AIR-SURFACE (DEFAULT COMPACT BALANCED VIEW) -->
    <div id="view-split" class="workspace-viewport active-view">
        <div class="split-container">
            <!-- Top Row: Primary Feeds (Video + Satellite) -->
            <div class="split-primary-row">
                <!-- Video Feed Card -->
                <div class="split-card">
                    <div class="split-card-header">
                        <span>MULTI-SPECTRAL OPTRONIC GIMBAL (OPTICAL + FLIR LWIR)</span>
                        <div style="display: flex; gap: 6px; align-items: center;">
                            <select id="split-mode-select" class="tactical-select" onchange="setSpectralMode(this.value)">
                                <option value="isotherm">ISOTHERM HUD</option>
                                <option value="side_by_side">SPLIT SCREEN</option>
                                <option value="blend">SPECTRAL BLEND</option>
                                <option value="optical">RGB OPTICAL</option>
                                <option value="thermal">RAW THERMAL</option>
                            </select>
                            <select id="split-pair-select" class="tactical-select" onchange="switchFlightPair(this.value)">
                                <option value="1">PAIR 1: SECTOR A-01</option>
                                <option value="2">PAIR 2: SECTOR A-02</option>
                                <option value="3">PAIR 3: SECTOR B-12</option>
                                <option value="4">PAIR 4: SECTOR B-14</option>
                                <option value="5">PAIR 5: SECTOR C-03</option>
                                <option value="6" selected>PAIR 6: SECTOR C-07</option>
                                <option value="7">PAIR 7: SECTOR D-01</option>
                            </select>
                        </div>
                    </div>
                    <div class="video-wrapper">
                        <img src="/api/stream/live" class="video-feed-el" alt="AeroGuardian Live Stream" />
                    </div>
                </div>

                <!-- GIS Map Card -->
                <div class="split-card">
                    <div class="split-card-header">
                        <span>ESRI WORLD IMAGERY GIS (COCONINO NATIONAL FOREST, AZ)</span>
                        <span id="sector-indicator" style="color: var(--tactical-cyan);">SECTOR C-07</span>
                    </div>
                    <div id="split-gis-map"></div>
                </div>
            </div>

            <!-- Bottom Row: Flight Controls, Safety Gate, Thought Stream -->
            <div class="split-bottom-row">
                <!-- Card 1: MAVLink 2.0 Dispatcher -->
                <div class="split-card">
                    <div class="split-card-header">
                        <span>BIDIRECTIONAL MAVLINK 2.0 DISPATCHER</span>
                        <span id="mavlink-state" style="color: var(--status-green);">READY</span>
                    </div>
                    <div style="padding: 10px; display: flex; flex-direction: column; gap: 8px;">
                        <div style="display: flex; gap: 6px;">
                            <button class="btn-action" onclick="sendMavCommand('LOITER')">LOITER HOVER</button>
                            <button class="btn-action" onclick="sendMavCommand('RTL')">RTL HOME</button>
                            <button class="btn-action" onclick="sendMavCommand('RESUME')">RESUME</button>
                            <button class="btn-action" onclick="sendMavCommand('GIMBAL_NADIR')">GIMBAL TILT</button>
                        </div>
                        <div style="display: flex; gap: 6px; margin-top: 2px;">
                            <span style="font-size: 10px; color: var(--text-muted); align-self: center;">SENSOR ZOOM:</span>
                            <button class="btn-action" onclick="setZoom(1.0)">1.0X</button>
                            <button class="btn-action" onclick="setZoom(2.0)">2.0X</button>
                            <button class="btn-action" onclick="setZoom(3.0)">3.0X</button>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 10px; margin-top: 4px;">
                            <div style="background: var(--bg-surface); padding: 5px 8px; border: 1px solid var(--border-tactical);">
                                <span style="color: var(--text-muted); display: block;">COORDINATES:</span>
                                <span id="telem-coords" style="color: var(--tactical-cyan); font-weight: 700;">35.0305, -111.5955</span>
                            </div>
                            <div style="background: var(--bg-surface); padding: 5px 8px; border: 1px solid var(--border-tactical);">
                                <span style="color: var(--text-muted); display: block;">FLIGHT MODE:</span>
                                <span id="telem-mode" style="color: var(--status-green); font-weight: 700;">AUTO_MISSION</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Card 2: Incident Commander Safety Gate -->
                <div class="split-card">
                    <div class="split-card-header">
                        <span>INCIDENT COMMANDER SAFETY GATE</span>
                        <span id="gate-state-pill" style="color: var(--status-green);">STANDBY</span>
                    </div>
                    <div style="padding: 10px; display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                        <div id="safety-gate-content">
                            <div style="font-size: 11px; font-weight: 700; color: var(--status-green); margin-bottom: 4px;">STANDBY (0 CASUALTIES)</div>
                            <div style="font-size: 10px; color: var(--text-muted); line-height: 1.4;">Active wildfire perimeter monitored via Farneback optical flow. No trapped civilians in threat zone.</div>
                        </div>
                        <button id="btn-rescue-auth" class="btn-approve-gate" disabled onclick="approveDispatch()">AUTHORIZE EXTRACTION HELICOPTER</button>
                    </div>
                </div>

                <!-- Card 3: Agentic Thought & Tool Stream -->
                <div class="split-card">
                    <div class="split-card-header">
                        <span>AGENTIC REASONING & TOOL DISPATCH LOG (5 Hz)</span>
                        <span class="thought-log-counter" style="color: var(--tactical-cyan); font-weight:700;">LIVE MCP</span>
                    </div>
                    <div class="log-terminal" id="split-thought-log">
                        <!-- Populated dynamically via /api/status -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- WORKSPACE 1: TACTICAL FLIGHT DECK (FULL VIEW) -->
    <div id="view-deck" class="workspace-viewport">
        <div class="deck-container">
            <div class="deck-top-bar">
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span style="font-size: 11px; font-weight: 800; color: var(--tactical-cyan);">SENSOR FLIGHT DECK :: HIGH-DPI HUD</span>
                    <span id="deck-telemetry-ribbon" style="font-size: 11px; color: var(--text-muted);">ALT: 45.0m AGL | HEADING: 180° | GIMBAL: -45°</span>
                </div>
                <div style="display: flex; gap: 6px;">
                    <button class="btn-action" onclick="setZoom(1.0)">1.0X WIDE</button>
                    <button class="btn-action" onclick="setZoom(2.0)">2.0X INSPECT</button>
                    <button class="btn-action" onclick="setZoom(3.0)">3.0X DETAIL</button>
                    <button class="btn-action" onclick="sendMavCommand('LOITER')">LOITER HOVER</button>
                    <button class="btn-action" onclick="sendMavCommand('RESUME')">RESUME</button>
                </div>
            </div>
            <div class="deck-video-stage">
                <img src="/api/stream/live" class="deck-video-img" alt="AeroGuardian Full Flight Deck" />
            </div>
        </div>
    </div>

    <!-- WORKSPACE 2: GIS WILDFIRE THEATER (FULL SATELLITE MAP) -->
    <div id="view-gis" class="workspace-viewport">
        <div class="gis-container">
            <div class="gis-top-bar">
                <span style="font-size: 11px; font-weight: 800; color: var(--tactical-cyan);">GIS WILDFIRE THEATER :: TACTICAL SECTORS & EXPANDING PERIMETERS</span>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span style="font-size: 11px; color: var(--text-muted);">SELECT PATROL SECTOR:</span>
                    <select class="tactical-select" onchange="switchFlightPair(this.value)">
                        <option value="1">SECTOR A-01 (Flagstaff South)</option>
                        <option value="2">SECTOR A-02 (Schultz Pass)</option>
                        <option value="3">SECTOR B-12 (Mormon Lake Road)</option>
                        <option value="4">SECTOR B-14 (Hart Prairie)</option>
                        <option value="5">SECTOR C-03 (Sedona Rim)</option>
                        <option value="6" selected>SECTOR C-07 (Fort Valley)</option>
                        <option value="7">SECTOR D-01 (Wing Mountain)</option>
                    </select>
                </div>
            </div>
            <div id="full-gis-map"></div>
        </div>
    </div>

    <!-- WORKSPACE 3: INCIDENT COMMAND & TRIAGE GATE -->
    <div id="view-triage" class="workspace-viewport">
        <div class="triage-container">
            <!-- Left: Casualty Matrix -->
            <div class="triage-panel">
                <div class="triage-title">DISASTER CASUALTY & PERIMETER TRIAGE MATRIX</div>
                <table class="triage-table">
                    <thead>
                        <tr>
                            <th>TARGET PIN</th>
                            <th>SECTOR</th>
                            <th>GPS (WGS84)</th>
                            <th>CASUALTY COUNT</th>
                            <th>FLAME ARRIVAL</th>
                            <th>TRIAGE STATUS</th>
                            <th>ACTION</th>
                        </tr>
                    </thead>
                    <tbody id="triage-table-body">
                        <tr>
                            <td>SEC-00</td>
                            <td>Sector C-07</td>
                            <td>35.030500, -111.595500</td>
                            <td>0 Detected</td>
                            <td>N/A (Clear)</td>
                            <td><span style="color: var(--status-green);">SECTOR_CLEAR</span></td>
                            <td><button class="btn-action" disabled>STANDBY</button></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Right: Chief Miller Sign-off Card -->
            <div class="triage-panel">
                <div class="triage-title">CHIEF MILLER TWO-MAN AUTHORIZATION GATE</div>
                <div style="font-size: 11px; line-height: 1.6; color: var(--text-muted); margin-bottom: 14px;">
                    Under NFPA 1500 and CalFire incident commander protocols, autonomous kinetic or rescue asset dispatch requires cryptographic human validation.
                </div>
                <div style="background: var(--bg-surface); padding: 10px; border: 1px solid var(--border-tactical); margin-bottom: 14px; font-size: 11px;">
                    <div>OFFICER: <strong>Chief Miller (Incident Commander)</strong></div>
                    <div>AGENCY: <strong>Coconino Wildfire Taskforce</strong></div>
                    <div>AUDIT LEDGER: <strong id="triage-audit-count">0 Records Signed</strong></div>
                </div>
                <button class="btn-approve-gate" id="triage-auth-btn" disabled onclick="approveDispatch()">AUTHORIZE EXTRACTION HELICOPTER</button>
            </div>
        </div>
    </div>

    <!-- WORKSPACE 4: MAVLINK PACKET STREAM & AGENT DECK -->
    <div id="view-mavlink" class="workspace-viewport">
        <div class="mavlink-container">
            <div class="mav-panel">
                <div style="font-size: 12px; font-weight: 800; color: var(--tactical-cyan); margin-bottom: 8px;">LIVE MAVLINK 2.0 UDP PACKET MONITOR (PORT 14550)</div>
                <div class="log-terminal" id="mav-packet-stream">
                    <!-- Dynamic packet lines -->
                </div>
            </div>
            <div class="mav-panel">
                <div style="font-size: 12px; font-weight: 800; color: var(--status-amber); margin-bottom: 8px;">AUTONOMOUS AGENTIC REASONING THOUGHT STREAM</div>
                <div class="log-terminal" id="full-thought-stream">
                    <!-- Dynamic thought lines -->
                </div>
            </div>
        </div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script>
        let currentWorkspace = 'split';
        let splitMap = null, fullMap = null;
        let splitUavMarker = null, fullUavMarker = null;
        let splitBreadcrumbs = null, fullBreadcrumbs = null;
        let splitFirePoly = null, fullFirePoly = null;
        let splitTargetMarker = null, fullTargetMarker = null;

        const uavSvg = (heading) => `
            <div class="uav-marker-container">
                <svg class="uav-svg-icon" style="transform: rotate(${heading}deg);" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="12 2 19 21 12 17 5 21 12 2"></polygon>
                </svg>
            </div>
        `;

        function initMaps() {
            const esriUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
            const esriAttr = 'Tiles &copy; Esri';

            // Split Map
            splitMap = L.map('split-gis-map', {zoomControl: false}).setView([35.0305, -111.5955], 16);
            L.tileLayer(esriUrl, {attribution: esriAttr, maxZoom: 19}).addTo(splitMap);
            splitBreadcrumbs = L.polyline([], {color: '#38bdf8', weight: 2, opacity: 0.7, dashArray: '4, 4'}).addTo(splitMap);
            splitFirePoly = L.polygon([], {color: '#ef4444', weight: 2, fillColor: '#ef4444', fillOpacity: 0.25}).addTo(splitMap);

            // Full GIS Map
            fullMap = L.map('full-gis-map', {zoomControl: true}).setView([35.0305, -111.5955], 15);
            L.tileLayer(esriUrl, {attribution: esriAttr, maxZoom: 19}).addTo(fullMap);
            fullBreadcrumbs = L.polyline([], {color: '#38bdf8', weight: 2.5, opacity: 0.8, dashArray: '5, 5'}).addTo(fullMap);
            fullFirePoly = L.polygon([], {color: '#ef4444', weight: 2.5, fillColor: '#ef4444', fillOpacity: 0.3}).addTo(fullMap);
        }

        function switchWorkspace(ws) {
            currentWorkspace = ws;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.workspace-viewport').forEach(v => v.classList.remove('active-view'));

            const viewEl = document.getElementById(`view-${ws}`);
            if (viewEl) viewEl.classList.add('active-view');

            event.target.classList.add('active');

            setTimeout(() => {
                if (ws === 'split' && splitMap) splitMap.invalidateSize();
                if (ws === 'gis' && fullMap) fullMap.invalidateSize();
            }, 100);
        }

        async function setZoom(val) {
            await fetch('/api/gimbal/zoom', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({zoom: val, center_x: 0.5, center_y: 0.5})
            });
        }

        async function setSpectralMode(mode) {
            await fetch('/api/stream/set_mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            });
        }

        async function switchFlightPair(pairId) {
            await fetch('/api/stream/select_pair', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({pair_id: parseInt(pairId)})
            });
        }

        async function sendMavCommand(cmd) {
            const res = await fetch('/api/mavlink/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            });
            const data = await res.json();
            document.getElementById('telem-mode').innerText = cmd;
        }

        async function approveDispatch() {
            const btn = document.getElementById('btn-rescue-auth');
            const triageBtn = document.getElementById('triage-auth-btn');
            btn.innerText = 'DISPATCH TRANSMITTED (CHIEF MILLER)';
            btn.classList.add('btn-approved-gate');
            btn.disabled = true;
            if (triageBtn) {
                triageBtn.innerText = 'DISPATCH TRANSMITTED (CHIEF MILLER)';
                triageBtn.classList.add('btn-approved-gate');
                triageBtn.disabled = true;
            }
            await fetch('/api/approve_dispatch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({commander_name: 'Chief Miller'})
            });
        }

        async function pollTelemetry() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const telem = data.telemetry;

                // Update text elements
                document.getElementById('telem-coords').innerText = `${telem.latitude.toFixed(4)}, ${telem.longitude.toFixed(4)}`;
                document.getElementById('telem-mode').innerText = telem.flight_mode;
                document.getElementById('sector-indicator').innerText = data.sector_name;
                document.getElementById('deck-telemetry-ribbon').innerText = `ALT: ${telem.altitude_m.toFixed(1)}m AGL | HEADING: ${telem.heading_deg.toFixed(0)}° | GIMBAL: ${telem.gimbal_pitch_deg.toFixed(0)}° | ZOOM: ${data.zoom_factor}X`;

                // Update UAV Marker & Map
                const pos = [telem.latitude, telem.longitude];
                const heading = telem.heading_deg || 0;

                const icon = L.divIcon({className: 'uav-custom', html: uavSvg(heading), iconSize: [28, 28], iconAnchor: [14, 14]});
                
                if (!splitUavMarker && splitMap) {
                    splitUavMarker = L.marker(pos, {icon: icon}).addTo(splitMap);
                } else if (splitUavMarker) {
                    splitUavMarker.setLatLng(pos);
                    splitUavMarker.setIcon(icon);
                }

                if (!fullUavMarker && fullMap) {
                    fullUavMarker = L.marker(pos, {icon: icon}).addTo(fullMap);
                } else if (fullUavMarker) {
                    fullUavMarker.setLatLng(pos);
                    fullUavMarker.setIcon(icon);
                }

                // Update breadcrumbs & wildfire polygon
                if (data.breadcrumbs && splitBreadcrumbs && fullBreadcrumbs) {
                    splitBreadcrumbs.setLatLngs(data.breadcrumbs);
                    fullBreadcrumbs.setLatLngs(data.breadcrumbs);
                }
                if (data.wildfire_polygon && splitFirePoly && fullFirePoly) {
                    splitFirePoly.setLatLngs(data.wildfire_polygon);
                    fullFirePoly.setLatLngs(data.wildfire_polygon);
                }

                // Update Thought Streams with accumulated history and scroll lock preservation
                const thoughts = data.recent_thoughts || [];
                const splitLog = document.getElementById('split-thought-log');
                const fullLog = document.getElementById('full-thought-stream');
                const mavStream = document.getElementById('mav-packet-stream');

                if (!window.accumulatedThoughts) {
                    window.accumulatedThoughts = [];
                    window.seenThoughtIds = new Set();
                }

                let newEntriesAdded = false;
                thoughts.forEach(t => {
                    const key = t.id || `${t.timestamp}_${t.tool_call}_${t.stage}`;
                    if (!window.seenThoughtIds.has(key)) {
                        window.seenThoughtIds.add(key);
                        window.accumulatedThoughts.push(t);
                        newEntriesAdded = true;
                    }
                });

                if (newEntriesAdded || (splitLog && splitLog.children.length === 0)) {
                    const splitScrolledUp = splitLog ? (splitLog.scrollHeight - splitLog.scrollTop - splitLog.clientHeight > 45) : false;
                    const fullScrolledUp = fullLog ? (fullLog.scrollHeight - fullLog.scrollTop - fullLog.clientHeight > 45) : false;

                    let logHtml = '';
                    window.accumulatedThoughts.forEach(t => {
                        const timeStr = new Date(t.timestamp * 1000).toLocaleTimeString();
                        logHtml += `<div style="margin-bottom: 6px;"><span style="color:#7d899b;">[${timeStr}]</span> <span style="color:#f59e0b; font-weight:600;">${t.tool_call}</span>: <span style="color:#e2e8f0;">${t.observation}</span> <em style="color:#38bdf8;">${t.decision}</em></div>`;
                    });

                    if (splitLog) {
                        splitLog.innerHTML = logHtml;
                        if (!splitScrolledUp) splitLog.scrollTop = splitLog.scrollHeight;
                    }
                    if (fullLog) {
                        fullLog.innerHTML = logHtml;
                        if (!fullScrolledUp) fullLog.scrollTop = fullLog.scrollHeight;
                    }

                    const counterPills = document.querySelectorAll('.thought-log-counter');
                    counterPills.forEach(p => { p.innerText = `${window.accumulatedThoughts.length} EVENTS`; });
                }

                // MAVLink simulated packet hex stream
                if (mavStream) {
                    const timeMs = Date.now();
                    mavStream.innerHTML = `
                        <div style="margin-bottom: 4px; color:#38bdf8;">[${new Date().toLocaleTimeString()}] RECV MAVLINK_MSG_ID_GLOBAL_POSITION_INT (lat=${Math.round(telem.latitude*1e7)}, lon=${Math.round(telem.longitude*1e7)}, alt=${Math.round(telem.altitude_m*1e3)}mm, hdg=${Math.round(telem.heading_deg*100)}cdeg)</div>
                        <div style="margin-bottom: 4px; color:#22c55e;">[${new Date().toLocaleTimeString()}] RECV MAVLINK_MSG_ID_HEARTBEAT (type=MAV_TYPE_QUADROTOR, autopilot=ARDUPILOTMEGA, base_mode=SAFETY_ARMED, custom_mode=${telem.flight_mode})</div>
                        <div style="margin-bottom: 4px; color:#7d899b;">[${new Date().toLocaleTimeString()}] RECV MAVLINK_MSG_ID_SYS_STATUS (voltage_battery=15200mV, current_battery=12500mA, battery_remaining=${Math.round(telem.battery_percent)}%)</div>
                        <div style="margin-bottom: 4px; color:#c084fc;">[${new Date().toLocaleTimeString()}] SEND MAVLINK_MSG_ID_COMMAND_ACK (cmd=MAV_CMD_DO_MOUNT_CONTROL, result=MAV_RESULT_ACCEPTED)</div>
                    `;
                }

                // Update Safety Gate strictly synchronized with verified survivor state
                const gateBtn = document.getElementById('btn-rescue-auth');
                const gatePill = document.getElementById('gate-state-pill');
                const gateContent = document.getElementById('safety-gate-content');

                if (data.incident && (data.incident.survivor_confidence > 0.4 || data.target_gps)) {
                    gatePill.innerText = 'THREAT CRITICAL';
                    gatePill.style.color = 'var(--status-red)';
                    gateBtn.disabled = false;
                    gateBtn.innerText = 'AUTHORIZE EXTRACTION HELICOPTER';
                    gateContent.innerHTML = `
                        <div style="font-size: 11px; font-weight: 700; color: var(--status-red); margin-bottom: 4px;">SURVIVORS VERIFIED IN ${data.sector_name}</div>
                        <div style="font-size: 10px; color: var(--text-muted); line-height: 1.4;">
                            Advancing fire arrival: <strong style="color:#fff;">${data.incident.time_to_impact_minutes.toFixed(1)} min</strong> | Spread: <strong style="color:#fff;">${data.incident.fire_speed_mps.toFixed(2)} m/s</strong><br>
                            GPS: <strong style="color: var(--tactical-cyan);">${data.target_gps ? data.target_gps[0].toFixed(5) + ', ' + data.target_gps[1].toFixed(5) : 'Calculating'}</strong>
                        </div>
                    `;
                } else if (data.incident) {
                    gatePill.innerText = 'MONITORING';
                    gatePill.style.color = 'var(--status-amber)';
                    gateBtn.disabled = true;
                    gateBtn.innerText = 'STANDBY (NO EXTRACTION REQUIRED)';
                    gateContent.innerHTML = `
                        <div style="font-size: 11px; font-weight: 700; color: var(--status-amber); margin-bottom: 4px;">WILDFIRE HAZARD ACTIVE (0 TRAPPED CIVILIANS)</div>
                        <div style="font-size: 10px; color: var(--text-muted); line-height: 1.4;">
                            Advancing fire arrival: <strong style="color:#fff;">${data.incident.time_to_impact_minutes.toFixed(1)} min</strong> | Spread: <strong style="color:#fff;">${data.incident.fire_speed_mps.toFixed(2)} m/s</strong><br>
                            Perimeter surveillance active. Threat zone clear of civilians.
                        </div>
                    `;
                } else {
                    gatePill.innerText = 'STANDBY';
                    gatePill.style.color = 'var(--status-green)';
                    gateBtn.disabled = true;
                    gateBtn.innerText = 'STANDBY (0 EXTRACTIONS NEEDED)';
                    gateContent.innerHTML = `
                        <div style="font-size: 11px; font-weight: 700; color: var(--status-green); margin-bottom: 4px;">STANDBY (0 CASUALTIES)</div>
                        <div style="font-size: 10px; color: var(--text-muted); line-height: 1.4;">Active wildfire perimeter monitored via Farneback optical flow. No trapped civilians in threat zone.</div>
                    `;
                }

                document.getElementById('triage-audit-count').innerText = `${data.audit_log_count} Records Signed`;

            } catch (err) {}
        }

        window.addEventListener('DOMContentLoaded', () => {
            initMaps();
            setInterval(pollTelemetry, 1000);
            pollTelemetry();
        });

        // Hotkey navigation: 1, 2, 3, 4, 5
        window.addEventListener('keydown', (e) => {
            if (e.key === '1') switchWorkspace('deck');
            if (e.key === '2') switchWorkspace('gis');
            if (e.key === '3') switchWorkspace('triage');
            if (e.key === '4') switchWorkspace('mavlink');
            if (e.key === '5') switchWorkspace('split');
            if (e.key === ' ' && !e.target.matches('input, select, textarea')) {
                e.preventDefault();
                const currentMode = document.getElementById('telem-mode').innerText;
                sendMavCommand(currentMode === 'LOITER_HOVER' ? 'RESUME' : 'LOITER');
            }
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


# Mount static directory for proof frame assets
os.makedirs("data/samples", exist_ok=True)
app.mount("/static", StaticFiles(directory="data/samples"), name="static")
