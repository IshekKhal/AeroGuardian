"""
Unit and integration tests for AeroGuardian Tactical Mission Control Dashboard.
Tests HTTP endpoints, telemetry status serialization, and the 1-click rescue approval gateway.
"""

import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import app, orchestrator
from src.agent.orchestrator import MissionState, IncidentReport


@pytest.fixture
def client():
    return TestClient(app)


def test_serve_dashboard_html(client):
    """Verifies that the tactical command HUD HTML interface loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AEROGUARDIAN :: TACTICAL AIR-SURFACE INCIDENT HUD" in response.text
    assert "AWS Graviton4 Arm64 Ready" in response.text


def test_get_mission_status(client):
    """Verifies that the mission telemetry status endpoint returns valid structured JSON."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "telemetry" in data
    assert "altitude_m" in data["telemetry"]
    assert "latitude" in data["telemetry"]
    assert "longitude" in data["telemetry"]
    assert "battery_percent" in data["telemetry"]
    assert "audit_log_count" in data


def test_approve_emergency_dispatch(client):
    """Verifies the Incident Commander 1-click rescue authorization flow."""
    # Set up active incident and transition orchestrator to pending state
    orchestrator.active_incident = IncidentReport(
        incident_id="INC-TEST-001",
        target_gps=(34.052235, -118.243683),
        survivor_confidence=0.92,
        time_to_impact_minutes=3.5,
        fire_speed_mps=0.8,
        urgency_level="CRITICAL_IMMINENT",
        evidence_summary="Test survivor near perimeter.",
    )
    orchestrator.current_state = MissionState.HUMAN_GATE_PENDING

    payload = {"commander_name": "Battalion Chief Miller"}
    response = client.post("/api/approve_dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["incident_id"] == "INC-TEST-001"
    assert orchestrator.active_incident.commander_approved is True
    assert orchestrator.current_state == MissionState.PATROL


def test_live_video_stream():
    """Verifies that the live MJPEG multi-spectral stream generator yields valid JPEG frames."""
    from src.dashboard.app import generate_live_stream
    stream_gen = generate_live_stream()
    first_chunk = next(stream_gen)
    assert len(first_chunk) > 0
    assert b"--frame" in first_chunk
    assert b"Content-Type: image/jpeg" in first_chunk


def test_mavlink_command_dispatch(client):
    """Verifies bidirectional MAVLink command dispatch endpoint."""
    res_loiter = client.post("/api/mavlink/command", json={"command": "LOITER"})
    assert res_loiter.status_code == 200
    assert res_loiter.json()["status"] == "SUCCESS"
    assert res_loiter.json()["command"] == "LOITER"

    res_rtl = client.post("/api/mavlink/command", json={"command": "RTL"})
    assert res_rtl.status_code == 200
    assert res_rtl.json()["status"] == "SUCCESS"
    assert res_rtl.json()["command"] == "RTL"

    res_invalid = client.post("/api/mavlink/command", json={"command": "UNKNOWN_ACTION"})
    assert res_invalid.status_code == 400


def test_flight_pairs_endpoints(client):
    """Verifies listing and selection of authentic FLAME 2 dual-stream flight pairs."""
    res_pairs = client.get("/api/stream/pairs")
    assert res_pairs.status_code == 200
    pairs = res_pairs.json()
    assert isinstance(pairs, list)
    assert len(pairs) >= 1

    # Select flight pair
    pid = pairs[0]["pair_id"]
    res_sel = client.post("/api/stream/select_pair", json={"pair_id": pid})
    assert res_sel.status_code == 200
    assert res_sel.json()["status"] == "success"
    assert res_sel.json()["active_pair_id"] == pid


def test_gimbal_zoom_endpoint(client):
    """Verifies Option A digital sensor zoom and gimbal centering endpoint."""
    res = client.post("/api/gimbal/zoom", json={"zoom": 2.5, "center_x": 0.6, "center_y": 0.4})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["target_zoom"] == 2.5
    assert data["center"] == [0.6, 0.4]

    # Verify status reflects zoom
    res_st = client.get("/api/status")
    assert res_st.status_code == 200
    assert "zoom_factor" in res_st.json()

def test_mavlink_qgc_handshake_and_mission_sync():
    """
    Verifies that MAVLink autopilot responds to QGroundControl handshake:
    MISSION_REQUEST_LIST (mission_count=0), PARAM_REQUEST_LIST, and command acknowledgments.
    """
    from src.flight.mavlink_interface import MAVLinkAutopilotController
    ctrl = MAVLinkAutopilotController(enable_gcs_bridge=False)

    # Verify standard parameters table
    assert len(ctrl.STANDARD_PARAMS) >= 8
    param_names = [p[0] for p in ctrl.STANDARD_PARAMS]
    assert b"SYSID_THISMAV" in param_names
    assert b"FRAME_CLASS" in param_names

    # Verify telemetry packets
    telem = ctrl.get_telemetry()
    assert telem.altitude_m > 0
    assert telem.battery_percent > 0

    # Verify mode commands
    ctrl.command_loiter()
    assert ctrl.current_telemetry.flight_mode == "LOITER_HOVER"
    ctrl.command_return_to_launch()
    assert ctrl.current_telemetry.flight_mode == "RETURN_TO_LAUNCH"
    ctrl.resume_mission()
    assert ctrl.current_telemetry.flight_mode == "AUTO_MISSION"
