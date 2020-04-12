import threading
import time
import math
import cv2
from CameraUtil import Video
from pymavlink import mavutil, rotmat

video = Video()
master = None
setpoint_msg = None
frame = None
red_thresholded = None


def getdronepositiondata():
    rotation_matrix = rotmat.Matrix3()
    while True:
        msg = master.recv_match()
        if not msg:
            continue
        elif msg.get_type() == 'ATTITUDE':
            msg = msg.to_dict()
            rotation_matrix.from_euler(msg['roll'], msg['pitch'], msg['yaw'])
            rotation_matrix = rotation_matrix.transposed()
            # print(rotation_matrix)

def send_current_setpoint_loop():
    """
    Sends the current setpoint message at 100Hz
    :return:
    """
    global setpoint_msg
    t = threading.Timer(1/100, send_current_setpoint_loop)
    t.daemon = True
    t.start()
    global setpoint_msg
    if setpoint_msg is not None:
        master.mav.send(setpoint_msg)


def init():
    global setpoint_msg, master
    master = mavutil.mavlink_connection("udpin:0.0.0.0:14540", baud=115200)

    print("Waiting for heartbeat")
    master.wait_heartbeat()
    print("Got heartbeat")

    print("Arming drone")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0)
    print("Drone armed")

    print("Setting initial setpoint")
    # Set velocity to 0
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )

    print("Starting offboard")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        29, 6, 0, 0, 0, 0, 0, 0
    )

    # Set position to 0, 0, -10 in NED frame
    setpoint_msg = master.mav.set_position_target_local_ned_encode(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000101111111000,
        0, 0, -10,
        0, 0, 0,
        0, 0, 0,
        1.517, 0
    )

    print("Waiting for first video frame")

    while not video.frame_available():
        continue

    print("Got first video frame")

    while True:
        msg = master.recv_match()
        if not msg:
            continue
        if msg.get_type() == 'LOCAL_POSITION_NED':
            msg = msg.to_dict()
            if math.fabs(msg['x']) < 0.5 and math.fabs(msg['y']) and math.fabs(msg['z'] + 10) < 0.5:
                break

def run():
    global frame, setpoint_msg
    send_current_setpoint_loop()
    print("Started send_current_setpoint_loop thread")
    init()
    print("Initialized")
    t = threading.Thread(target=getdronepositiondata, args=())
    t.daemon = True
    t.start()

    print("Started")
    while True:
        frame = video.frame()
        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Sending new waypoint")
            # Set position to 0, 5, -10 in NED frame
            setpoint_msg = master.mav.set_position_target_local_ned_encode(
                0,
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000101111111000,
                0, 5, -10,
                0, 0, 0,
                0, 0, 0,
                1.517, 0
            )
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print("Sending new waypoint")
            # Set position to 0, 5, -10 in NED frame
            setpoint_msg = master.mav.set_position_target_local_ned_encode(
                0,
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000101111111000,
                0, 0, -10,
                0, 0, 0,
                0, 0, 0,
                1.517, 0
            )

    t.join()


if __name__ == "__main__":
    run()
    # while True:
    #     cv2.imshow("rgb", frame)
    #     cv2.imshow("red thresholded", red_thresholded)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
