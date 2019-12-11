import asyncio
import threading
import time
import math
import cv2
from CameraUtil import Video
from pymavlink import mavutil
import csv

# Some setup
gd = {
    'time': "",
    'x_bb': "",
    'y_bb': "",
    'w_bb': "",
    'h_bb': "",
    'f_u': 0.5,
    'f_v': 0.5,
    'f_delta': 0.0,
    'delta_f_u_psi': 0.0,
    'delta_f_u_y': 0.0,
    'delta_f_v_z': 0.0,
    'delta_f_delta_x': 0.0,
    'delta_x_tme': 0.0,
    'delta_y_tme': 0.0,
    'delta_psi_tme': 0.0,
    'delta_z_tme': 0.0,
    'prev_delta_x_tme': 0.0,
    'prev_delta_y_tme': 0.0,
    'prev_delta_psi_tme': 0.0,
    'prev_delta_z_tme': 0.0,
    'v_xr': "",
    'v_yr': "",
    'v_zr': "",
    'yawrate': "",
    'pitch_angle': 0.0,
    'roll_angle': 0.0,
    'yaw_angle': 90.0,
    'w_im': 1280,
    'h_im': 720,
    'alpha_u': "",
    'alpha_b': "",
    'got_initial_frame': False,
    'initial_f_u': 0.5,
    'initial_f_v': 0.5,
    'initial_f_delta': 13,
    'psi_telem_ref': 90.0,
    'psi_telem': None,
    'theta_centroid_ref': 0.0,
    'theta_centroid': None,
    'FOV_u': 90.0,
    'FOV_v': "",
    'A_exp': 1225,  # m or cm or px?
    'd_exp': 3.0,
    'kp_vx': 0.0254,
    'kd_vx': 0.0124,
    'kp_vy': 0.298,
    'kd_vy': 0.145,
    'kp_yaw': 0.990,
    'kd_yaw': 0.119,
    'kp_vz': 1.430,
    'kd_vz': 0.371,
    'camera_period': 1/30,
    'controller_period': 1/100,
    'v_xlat': None,
    'v_ylon': None,
    'v_zalt': None,
    'local_vx': None,
    'local_vy': None,
    'local_vz': None
}

video = Video()
master = None
setpoint_msg = None


def getdronepositiondata():
    while True:
        msg = master.recv_match()
        if not msg:
            continue
        if msg.get_type() == 'LOCAL_POSITION_NED_COV':
            msg = msg.to_dict()
            gd['local_x'] = msg['lat']
            gd['local_y'] = msg['lon']
            gd['local_z'] = msg['alt']
            gd['local_vx'] = msg['vx']
            gd['local_vy'] = msg['vy']
            gd['local_vz'] = msg['vz']


def _gettargetposition():
    """
    This is called by getcentroidata()
    :return:
    """
    # while not video.frame_available():
    #     continue

    frame = video.frame()
    red = frame[:, :, 2]
    # green = frame[:, :, 1]
    # blue = frame[:, :, 0]
    _, red_thresholded = cv2.threshold(red, 120, 255, cv2.THRESH_BINARY)
    # _, green_thresholded = cv2.threshold(green, 120, 255, cv2.THRESH_BINARY)
    # _, blue_thresholded = cv2.threshold(blue, 120, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(red_thresholded, 1, 2)
    if len(contours) == 0:
        return None, None, None, None

    largest_cnt_area = 0
    largest_cnt_idx = 0
    cnt_idx = 0
    for c in contours:
        if cv2.contourArea(c) > largest_cnt_area:
            largest_cnt_area = cv2.contourArea(c)
            largest_cnt_idx = cnt_idx
        cnt_idx = cnt_idx + 1
    x, y, w, h = cv2.boundingRect(contours[largest_cnt_idx])
    cv2.rectangle(red_thresholded, (x, y), (x + w, y + h), (255, 255, 255))

    # titles = ["rgb", "red", "green", "blue"]
    # images = [frame, red_thresholded, green_thresholded, blue_thresholded]
    cv2.imshow('rgb', frame)
    cv2.imshow('red', red_thresholded)
    # cv2.imshow('green', green_thresholded)
    # cv2.imshow('blue', blue_thresholded)

    # print(red_thresholded.item(80, 100))
    print("x_bb: %3d y_bb: %3d w_bb: %3d h_bb: %3d" % (x, y, w, h))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return x

    gd['x_bb'] = x
    gd['y_bb'] = y
    gd['w_bb'] = w
    gd['h_bb'] = h

    return x, y, w, h


def getcentroiddata():
    """
    Calculates centroid of target in camera frame. Runs at 30Hz
    :return:
    """
    while True:
        starttime = time.time()
        x_bb, y_bb, w_bb, h_bb = _gettargetposition()
        w_im = gd['w_im']
        h_im = gd['h_im']

        if x_bb is None:  # if nothing is in frame, don't send any information
            return None, None, None

        if (x_bb + w_bb == w_im or  # if target is at edge, don't send any information
                x_bb == 0 or
                y_bb + h_bb == h_im or
                y_bb == 0):
            return None, None, None

        # Equation 1 from Pestana "Computer vision based general object following
        f_u = (x_bb + (w_bb / 2)) / w_im
        f_v = (y_bb + (h_bb / 2)) / h_im
        f_delta = math.sqrt((w_im * h_im) / (w_bb * h_bb))

        gd['f_u'] = f_u
        gd['f_v'] = f_v
        gd['f_delta'] = f_delta

        while time.time() - starttime < 1/30:
            continue


def _decouplecentroiddata():
    """
    Decouples the attitude of the quadrotor from the camera featueres. Called by getsetpoints()
    :return:
    """
    global gd

    f_u = gd['f_u']
    f_v = gd['f_v']
    f_delta = gd['f_delta']
    initial_f_u = gd['initial_f_u']
    initial_f_v = gd['initial_f_v']
    initial_f_delta = gd['initial_f_delta']
    psi_telem_ref = gd['psi_telem_ref']
    theta_centroid_ref = gd['theta_centroid_ref']
    FOV_u = gd['FOV_u']
    FOV_v = gd['FOV_v']

    if f_u is None:
        return None, None, None, None

    # Equation 2 from Pestana "Computer vision based general object following
    delta_f_u_psi = f_u - initial_f_u
    delta_f_u_y = delta_f_u_psi - ((psi_telem_ref - gd['yaw_angle']) / FOV_u)
    delta_f_v_z = (f_v - initial_f_v) - ((theta_centroid_ref - gd['pitch_angle']) / FOV_v)
    delta_f_delta_x = f_delta - initial_f_delta

    gd['delta_f_u_psi'] = delta_f_u_psi
    gd['delta_f_u_y'] = delta_f_u_y
    gd['delta_f_v_z'] = delta_f_v_z
    gd['delta_f_delta_x'] = delta_f_delta_x


def getsetpoints():
    """
    Calculates setpoints. Runs at 100Hz
    :return:
    """
    global gd
    while True:
        starttime = time.time()
        _decouplecentroiddata()
        A_exp = gd['A_exp']
        d_exp = gd['d_exp']
        w_im = gd['w_im']
        h_im = gd['h_im']
        alpha_u = gd['alpha_u']
        alpha_v = gd['alpha_v']
        prev_delta_psi_tme = gd['prev_delta_psi_tme']
        prev_delta_x_tme = gd['prev_delta_x_tme']
        prev_delta_y_tme = gd['prev_delta_y_tme']
        prev_delta_z_tme = gd['prev_delta_z_tme']
        kp_vx = gd['kp_vx']
        kd_vx = gd['kd_vx']
        kp_vy = gd['kp_vy']
        kd_vy = gd['kd_vy']
        kp_yaw = gd['kp_yaw']
        kd_yaw = gd['kd_yaw']
        kp_vz = gd['kp_vz']
        kd_vz = gd['kd_vz']
        camera_period = gd['camera_period']
        controller_period = gd['controller_period']
        FOV_u = gd['FOV_u']
        delta_f_u_psi = gd['delta_f_u_psi']
        delta_f_u_y = gd['delta_f_u_y']
        delta_f_v_z = gd['delta_f_v_z']
        delta_f_delta_x = gd['delta_f_delta_x']

        if delta_f_u_psi is None:
            gd['v_xr'] = None
            gd['v_yr'] = None
            gd['yawrate'] = None
            gd['v_zr'] = None
            return

        # x velocity controller
        delta_x_tme = delta_f_delta_x * math.sqrt(A_exp) * math.sqrt((alpha_u * alpha_v) / (w_im * h_im))
        v_xr = delta_x_tme * kp_vx + ((delta_x_tme - prev_delta_x_tme) * (controller_period)) * kd_vx
        prev_delta_x_tme = delta_x_tme

        # y velocity controller
        delta_y_tme = delta_f_u_y * d_exp * (w_im / alpha_u)
        v_yr = delta_y_tme * kp_vy + ((delta_y_tme - prev_delta_y_tme) * (controller_period)) * kd_vy
        prev_delta_y_tme = delta_y_tme

        # yawrate controller
        delta_psi_tme = delta_f_u_psi * FOV_u
        yawrate = delta_psi_tme * kp_yaw + ((delta_psi_tme - prev_delta_psi_tme) * (controller_period)) * kd_yaw
        prev_delta_psi_tme = delta_psi_tme

        # z velocity controller
        delta_z_tme = delta_f_v_z * d_exp * (h_im / alpha_v)
        v_zr = delta_z_tme * kp_vz + ((delta_z_tme - prev_delta_z_tme) * (controller_period)) * kd_vz
        prev_delta_z_tme = delta_z_tme

        gd['delta_x_tme'] = delta_x_tme
        gd['prev_delta_x_tme'] = prev_delta_x_tme
        gd['v_xr'] = v_xr
        gd['delta_y_tme'] = delta_y_tme
        gd['prev_delta_y_tme'] = prev_delta_y_tme
        gd['v_yr'] = v_yr
        gd['delta_psi_tme'] = delta_psi_tme
        gd['prev_delta_psi_tme'] = prev_delta_psi_tme
        gd['yawrate'] = yawrate
        gd['delta_z_tme'] = delta_z_tme
        gd['prev_delta_z_tme'] = prev_delta_z_tme
        gd['v_zr'] = v_zr

        while time.time() - starttime < 1/100:
            continue


def send_current_setpoint_loop():
    """
    Sends the current setpoint message at 100Hz
    :return:
    """
    global setpoint_msg
    while setpoint_msg is None:
        continue
    while True:
        starttime = time.time()
        master.mav.send(setpoint_msg)
        while time.time() - starttime < 1/100:
            continue


def init():
    global gd, master, setpoint_msg

    gd['alpha_u'] = gd['w_im']
    gd['alpha_v'] = gd['h_im']
    gd['FOV_v'] = gd['h_im'] / gd['w_im'] * gd['FOV_u']

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

    # Set position to 0, 0, 10
    setpoint_msg = master.mav.set_position_target_local_ned_encode(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,
        0, 0, -10,
        0, 0, 0,
        0, 0, 0,
        0, 0
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
    global gd, setpoint_msg

    threading.Thread(target=send_current_setpoint_loop, args=()).start()
    init()
    threading.Thread(target=getdronepositiondata, args=()).start()
    threading.Thread(target=getcentroiddata, args=()).start()
    threading.Thread(target=getsetpoints, args=()).start()

    _gettargetposition()

    input("Press Enter to start...")
    timeout_cnt = 0

    with open('debug_data.csv', mode='w') as csv_file:
        fieldnames = gd.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            gd['time'] = time.time()

            v_xr = gd['v_xr']
            v_yr = gd['v_yr']
            yawrate = gd['yawrate']
            v_zr = gd['v_zr']
            local_vx = gd['local_vx']
            local_vy = gd['local_vy']
            local_vz = gd['local_vz']

            # If target is not in camera frame for a while, stop moving.
            # Note: v_xr is None when camera doesn't see anything
            if v_xr is None:
                timeout_cnt = timeout_cnt + gd['controller_period']
                if timeout_cnt > 1/5:
                    # Set velocity to 0
                    setpoint_msg = master.mav.set_position_target_local_ned_encode(
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
            # Otherwise, keep going
            else:
                setpoint_msg = master.mav.set_position_target_local_ned_encode(
                    0,
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b0000101111000111,
                    0, 0, 0,
                    v_xr, v_yr, v_zr,
                    0, 0, 0,
                    0, yawrate
                )

            print("vxr:%5.3f\tvyr:%5.3f\tvzr:%5.3f\tvpsir:%5.3f\tvx:%5.3f\tvy:%5.3f\tvz:%5.3f"
                  % (v_xr, v_yr, v_zr, yawrate, local_vx, local_vy, local_vz))

            writer.writerow(gd)
            timeout_cnt = 0
            while time.time() - gd['time'] < gd['controller_period']:
                continue


if __name__ == "__main__":
    run()
