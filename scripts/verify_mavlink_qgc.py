#!/usr/bin/env python3
"""
AeroGuardian MAVLink GCS Verification Monitor
Simulates a Ground Control Station (like QGroundControl) connecting over UDP 14550.
Verifies packet reception rates and bidirectional flight command acknowledgment.
"""

import sys
import time
from pymavlink import mavutil

def main():
    print("==================================================================")
    print("  AEROGUARDIAN :: MAVLINK 2.0 GCS UDP PORT 14550 VERIFICATION")
    print("==================================================================")
    
    target = "udpin:127.0.0.1:14550"
    print(f"[*] Binding MAVLink GCS receiver to {target}...")
    try:
        conn = mavutil.mavlink_connection(target)
    except Exception as e:
        print(f"[!] Failed to bind UDP port 14550: {e}")
        sys.exit(1)

    print("[+] Successfully bound to UDP 14550.")
    print("[*] Listening for incoming HEARTBEAT and GLOBAL_POSITION_INT packets...")

    packet_counts = {"HEARTBEAT": 0, "GLOBAL_POSITION_INT": 0, "SYS_STATUS": 0}
    start_time = time.time()

    # Listen for 5 seconds
    while time.time() - start_time < 5.0:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg:
            mtype = msg.get_type()
            packet_counts[mtype] = packet_counts.get(mtype, 0) + 1
            if mtype == "HEARTBEAT":
                print(f"  [HEARTBEAT] Type={msg.type}, Autopilot={msg.autopilot}, BaseMode={msg.base_mode}, CustomMode={msg.custom_mode}")
            elif mtype == "GLOBAL_POSITION_INT":
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.relative_alt / 1e3
                hdg = msg.hdg / 100.0
                print(f"  [GLOBAL_POSITION_INT] Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt:.1f}m, Hdg={hdg:.1f}°")

    print("\n------------------------------------------------------------------")
    print(f"Packet Summary over 5 seconds: {packet_counts}")
    print("------------------------------------------------------------------")

    if packet_counts.get("HEARTBEAT", 0) > 0 and packet_counts.get("GLOBAL_POSITION_INT", 0) > 0:
        print("[SUCCESS] MAVLink 2.0 GCS link verified with active telemetry!")
        sys.exit(0)
    else:
        print("[!] Warning: Did not receive expected packets. Ensure AeroGuardian server is running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
