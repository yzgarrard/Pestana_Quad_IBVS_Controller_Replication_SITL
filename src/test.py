from pymavlink import mavutil
from pymavlink.dialects.v10 import common
import threading

con = mavutil.mavlink_connection(device="udp::14540")


def sendheartbeat():
    t = threading.Timer(1.0, sendheartbeat)
    t.daemon = True
    t.start()
    con.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


con.wait_heartbeat()

# Choose a mode
mode = 'OFFBOARD'

# Check if mode is available
if mode not in con.mode_mapping():
    print('Unknown mode : {}'.format(mode))
    print('Try:', list(con.mode_mapping().keys()))
    exit(1)

# con.mav.command_long_send(
#     con.target_system,
#     con.target_component,
#     mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
#     0,
#     1, 0, 0, 0, 0, 0, 0)
# print("Sent arming command")

con.mav.command_long_send(
    con.target_system,
    con.target_component,
    mavutil.mavlink.MAV_CMD_MISSION_START,
    0,
    0, 1, 0, 0, 0, 0, 0)
print("Sent mission start command")
