import asyncio
import threading
import time
import math
import cv2
from CameraUtil import Video
from pymavlink import mavutil, mavextra
import csv

# Some setup
gd = {
    'time': "",
    'x_bb': 0,
    'y_bb': 0,
    'w_bb': 0,
    'h_bb': 0,
    'w_im': 1280,
    'h_im': 720,
    'f_u': 0.5,
    'f_v': 0.5,
    'f_delta': 0.0,
    'psi_telem_ref': 90,
    'psi_telem': None,
    'theta_centroid_ref': 0.0,
    'theta_centroid': None,
    'delta_f_u_psi': 0.0,
    'delta_f_u_y': 0.0,
    'delta_f_v_z': 0.0,
    'delta_f_delta_x': 0.0,
    'A_exp': 1225,  # m or cm or px?
    'alpha_u': "",
    'alpha_v': "",
    'd_exp': 3.0,
    'FOV_u': 90.0,
    'FOV_v': "",
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
    'got_initial_frame': False,
    'f_u_ref': 0.5,
    'f_v_ref': 0.5,
    'f_delta_ref': 13,
    'kp_vx': 0.0254,
    'kd_vx': 0.0124,
    'kp_vy': -0.298,
    'kd_vy': -0.145,
    'kp_yaw': 0.990,
    'kd_yaw': 0.119,
    'kp_vz': 1.430,
    'kd_vz': 0.371,
    'camera_period': 1 / 30,
    'controller_period': 1 / 100,
    'v_xlat': 0,
    'v_ylon': 0,
    'v_zalt': 0,
    'local_x': 0,
    'local_y': 0,
    'local_z': 0,
    'local_vx': 0,
    'local_vy': 0,
    'local_vz': 0
}

video = Video()
master = None
setpoint_msg = None


def getdronepositiondata():
    while True:
        msg = master.recv_match()
        if not msg:
            continue
        if msg.get_type() == 'LOCAL_POSITION_NED':
            msg = msg.to_dict()
            gd['local_x'] = msg['x']
            gd['local_y'] = msg['y']
            gd['local_z'] = msg['z']
            gd['local_vx'] = msg['vx']
            gd['local_vy'] = msg['vy']
            gd['local_vz'] = msg['vz']
        elif msg.get_type() == 'ATTITUDE':
            msg = msg.to_dict()
            gd['roll_angle'] = math.degrees(msg['roll'])
            gd['pitch_angle'] = math.degrees(msg['pitch'])
            gd['yaw_angle'] = math.degrees(msg['yaw'])
            gd['psi_telem'] = math.degrees(msg['yaw'])
            gd['theta_centroid'] = math.degrees(msg['pitch'])


def _gettargetposition():
    """
    This is called by getcentroidata()
    :return:
    """
    global gd

    frame = video.frame()
    red = frame[:, :, 2]
    # green = frame[:, :, 1]
    # blue = frame[:, :, 0]
    _, red_thresholded = cv2.threshold(red, 120, 255, cv2.THRESH_BINARY)
    # _, green_thresholded = cv2.threshold(green, 120, 255, cv2.THRESH_BINARY)
    # _, blue_thresholded = cv2.threshold(blue, 120, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(red_thresholded, 1, 2)
    if len(contours) == 0:
        gd['x_bb'] = None
        gd['y_bb'] = None
        gd['w_bb'] = None
        gd['h_bb'] = None
        return

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
    # cv2.imshow('rgb', frame)
    # cv2.imshow('red', red_thresholded)
    # cv2.imshow('green', green_thresholded)
    # cv2.imshow('blue', blue_thresholded)

    # print(red_thresholded.item(80, 100))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return x

    gd['x_bb'] = x
    gd['y_bb'] = y
    gd['w_bb'] = w
    gd['h_bb'] = h
    return


def getcentroiddata():
    """
    Calculates centroid of target in camera frame. Runs at 30Hz
    :return:
    """
    t = threading.Timer(1 / 30, getcentroiddata)
    t.daemon = True
    t.start()
    global gd
    _gettargetposition()
    x_bb = gd['x_bb']
    y_bb = gd['y_bb']
    w_bb = gd['w_bb']
    h_bb = gd['h_bb']
    w_im = gd['w_im']
    h_im = gd['h_im']

    if x_bb is None:  # if nothing is in frame, don't send any information
        gd['f_u'] = None
        gd['f_v'] = None
        gd['f_delta'] = None
        return

    if (x_bb + w_bb == w_im or  # if target is at edge, don't send any information
            x_bb == 0 or
            y_bb + h_bb == h_im or
            y_bb == 0):
        gd['f_u'] = None
        gd['f_v'] = None
        gd['f_delta'] = None
        return

    # Equation 1 from Pestana "Computer vision based general object following
    f_u = (x_bb + (w_bb / 2)) / w_im
    f_v = (y_bb + (h_bb / 2)) / h_im
    f_delta = math.sqrt((w_im * h_im) / (w_bb * h_bb))

    gd['f_u'] = f_u
    gd['f_v'] = f_v
    gd['f_delta'] = f_delta
    return


def _decouplecentroiddata():
    """
    Decouples the attitude of the quadrotor from the camera featueres. Called by getsetpoints()
    :return:
    """
    global gd

    f_u = gd['f_u']
    f_v = gd['f_v']
    f_delta = gd['f_delta']
    initial_f_u = gd['f_u_ref']
    initial_f_v = gd['f_v_ref']
    initial_f_delta = gd['f_delta_ref']
    psi_telem_ref = gd['psi_telem_ref']
    theta_centroid_ref = gd['theta_centroid_ref']
    FOV_u = gd['FOV_u']
    FOV_v = gd['FOV_v']

    if f_u is None:
        gd['delta_f_u_psi'] = None
        gd['delta_f_u_y'] = None
        gd['delta_f_v_z'] = None
        gd['delta_f_delta_x'] = None
        return

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
    t = threading.Timer(1 / 100, getsetpoints)
    t.daemon = True
    t.start()
    global gd
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


def send_current_setpoint_loop():
    """
    Sends the current setpoint message at 100Hz
    :return:
    """
    t = threading.Timer(1 / 100, send_current_setpoint_loop)
    t.daemon = True
    t.start()
    global setpoint_msg
    if setpoint_msg is not None:
        master.mav.send(setpoint_msg)


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


def calculate_setpoint():
    t = threading.Timer(1 / 100, calculate_setpoint)
    t.daemon = True
    t.start()
    global gd, setpoint_msg
    with open('debug_data.csv', mode='a+') as csv_file:
        fieldnames = gd.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        gd['time'] = time.time()

        v_xr = gd['v_xr']
        v_yr = gd['v_yr']
        yawrate = gd['yawrate']
        v_zr = gd['v_zr']
        local_vz = gd['local_vz']
        yaw_angle = math.radians(gd['yaw_angle'])
        roll_angle = math.radians(gd['roll_angle'])
        pitch_angle = math.radians(gd['pitch_angle'])

        # If target is not in camera frame for a while, stop moving.
        # Note: v_xr is None when camera doesn't see anything
        if v_xr is None:
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
            # setpoint_msg = master.mav.set_position_target_local_ned_encode(
            #     0,
            #     master.target_system,
            #     master.target_component,
            #     mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            #     0b0000011111000111,
            #     0, 0, 0,
            #     v_xr, v_yr, v_zr,
            #     0, 0, 0,
            #     0, math.radians(yawrate)
            # )
            kp_z = 1  # this gain is separate from the controller we are replicating
            pitch_r = v_xr / 14.0  # desired pitch angle = desired velocity / (14m/s/rad)
            roll_r = v_yr / 14.0
            if pitch_r > 0.209:
                pitch_r = 0.209
            elif pitch_r < -0.209:
                pitch_r = -0.209
            if roll_r > 0.209:
                roll_r = 0.209
            elif roll_r < -0.209:
                roll_r = -0.209
            total_thrust = (1 / (math.cos(pitch_angle) * math.cos(roll_angle))) * (
                        kp_z * (math.pow(v_zr, 2))) + local_vz
            if total_thrust > 0.3:
                total_thrust = 0.3
            quat = mavextra.euler_to_quat([roll_r, pitch_r, yaw_angle]).tolist()

            print("%6.3f\t%6.3f\t%6.3f\t%6.3f\t" % (pitch_r, roll_r, v_zr, total_thrust))
            setpoint_msg = master.mav.set_attitude_target_encode(
                0,
                master.target_system,
                master.target_component,
                0b00000111,
                quat,
                0, 0, 0,
                total_thrust
            )

        # print(gd)
        writer.writerow(gd)


def run():
    global gd

    send_current_setpoint_loop()
    print("Started send_current_setpoint_loop thread")
    init()
    print("Initialized")
    t = threading.Thread(target=getdronepositiondata, args=())
    t.daemon = True
    t.start()
    print("Started getdronepositiondata thread")
    getcentroiddata()
    print("Started getcentroiddata")
    getsetpoints()

    input("Press Enter to start...")

    # Clear the data file
    with open('debug_data.csv', mode='w+') as csv_file:
        fieldnames = gd.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    calculate_setpoint()

    t.join()


if __name__ == "__main__":
    run()
