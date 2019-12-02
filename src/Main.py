import asyncio
import time
import math
import cv2
from CameraUtil import Video
from mavsdk import System
from mavsdk import (OffboardError, VelocityBodyYawspeed, PositionNedYaw, Telemetry)
import csv

# Some setup
w_im = 1280  # image width in pixels
h_im = 720  # image height in pixels
alpha_u = w_im
alpha_v = h_im
got_initial_frame = False
initial_f_u = None
initial_f_v = None
initial_f_delta = None
psi_telem_ref = 90.0
psi_telem = None
theta_centroid_ref = 0.0
theta_centroid = None
FOV_u = 90.0
FOV_v = (h_im / w_im) * FOV_u
A_exp = 1225  # m or cm or px?
d_exp = 3.0
prev_delta_x_tme = 0.0
prev_delta_y_tme = 0.0
prev_delta_psi_tme = 0.0
prev_delta_z_tme = 0.0

# kp_vx = 0.0254
# kd_vx = 0.0124
# kp_vy = -0.298
# kd_vy = -0.145
# kp_yaw = -0.990
# kd_yaw = -0.119
# kp_vz = 1.430
# kd_vz = 0.371

kp_vx = 0.0254
kd_vx = 0.0124
kp_vy = 0.298
kd_vy = 0.145
kp_yaw = 0.990
kd_yaw = 0.119
kp_vz = 1.430
kd_vz = 0.371

# kp_vx = .25
# kd_vx = 0.1
# kp_vy = 0
# kd_vy = 0
# kp_yaw = 1
# kd_yaw = 0.01
# kp_vz = 0.01
# kd_vz = 0.01

euler_angles = None
drone_position = None

controller_period = 1/30

csv_row = {
    'time': "",
    'x_bb': "",
    'y_bb': "",
    'w_bb': "",
    'h_bb': "",
    'f_u': "",
    'f_v': "",
    'f_delta': "",
    'delta_f_u_psi': "",
    'delta_f_u_y': "",
    'delta_f_v_z': "",
    'delta_f_delta_x': "",
    'delta_x_tme': "",
    'delta_y_tme': "",
    'delta_psi_tme': "",
    'delta_z_tme': "",
    'prev_delta_x_tme': "",
    'prev_delta_y_tme': "",
    'prev_delta_psi_tme': "",
    'prev_delta_z_tme': "",
    'v_xr': "",
    'v_yr': "",
    'v_zr': "",
    'yawrate': "",
    'pitch_angle': "",
    'roll_angle': "",
    'yaw_angle': ""
}

video = Video()


def gettargetposition():
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

    csv_row['x_bb'] = x
    csv_row['y_bb'] = y
    csv_row['w_bb'] = w
    csv_row['h_bb'] = h

    return x, y, w, h


def getcentroiddata():
    x_bb, y_bb, w_bb, h_bb = gettargetposition()

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

    csv_row['f_u'] = f_u
    csv_row['f_v'] = f_v
    csv_row['f_delta'] = f_delta

    return f_u, f_v, f_delta


def decouplecentroiddata():
    f_u, f_v, f_delta = getcentroiddata()
    global got_initial_frame, initial_f_u, initial_f_v, initial_f_delta, FOV_u, FOV_v

    if f_u is None:
        return None, None, None, None

    if not got_initial_frame:
        initial_f_u = 0.5  # f_u
        initial_f_v = 0.25  # f_v
        initial_f_delta = 13
        got_initial_frame = True

    # Equation 2 from Pestana "Computer vision based general object following
    delta_f_u_psi = f_u - initial_f_u
    delta_f_u_y = delta_f_u_psi - ((psi_telem_ref - euler_angles.yaw_deg) / FOV_u)
    delta_f_v_z = (f_v - initial_f_v) - ((theta_centroid_ref - euler_angles.pitch_deg) / FOV_v)
    delta_f_delta_x = f_delta - initial_f_delta

    csv_row['delta_f_u_psi'] = delta_f_u_psi
    csv_row['delta_f_u_y'] = delta_f_u_y
    csv_row['delta_f_v_z'] = delta_f_v_z
    csv_row['delta_f_delta_x'] = delta_f_delta_x

    return delta_f_u_psi, delta_f_u_y, delta_f_v_z, delta_f_delta_x


def getsetpoints():
    delta_f_u_psi, delta_f_u_y, delta_f_v_z, delta_f_delta_x = decouplecentroiddata()
    global A_exp, d_exp, w_im, h_im, alpha_u, alpha_v, prev_delta_psi_tme, prev_delta_x_tme, prev_delta_y_tme, prev_delta_z_tme
    global kp_vx, kd_vx, kp_vy, kd_vy, kp_yaw, kd_yaw, kp_vz, kd_vz

    if delta_f_u_psi is None:
        return None, None, None, None

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

    csv_row['delta_x_tme'] = delta_x_tme
    csv_row['prev_delta_x_tme'] = prev_delta_x_tme
    csv_row['v_xr'] = v_xr
    csv_row['delta_y_tme'] = delta_y_tme
    csv_row['prev_delta_y_tme'] = prev_delta_y_tme
    csv_row['v_yr'] = v_yr
    csv_row['delta_psi_tme'] = delta_psi_tme
    csv_row['prev_delta_psi_tme'] = prev_delta_psi_tme
    csv_row['yawrate'] = yawrate
    csv_row['delta_z_tme'] = delta_z_tme
    csv_row['prev_delta_z_tme'] = prev_delta_z_tme
    csv_row['v_zr'] = v_zr

    return v_xr, v_yr, yawrate, v_zr


async def run():
    global euler_angles
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone discovered with UUID: {state.uuid}")
            break

    async for is_armed in drone.telemetry.armed():
        print("Is_armed:", is_armed)
        if not is_armed:
            print("-- Arming")
            await drone.action.arm()
        break

    print("-- Setting initial setpoint")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
              {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -10.0, 90.0))
    async for drone_position in drone.telemetry.odometry():
        if (math.fabs(drone_position.position_body.x_m) < 0.5 and
                math.fabs(drone_position.position_body.y_m) < 0.5 and
                math.fabs(drone_position.position_body.z_m + 10) < 0.5):
            break

    while not video.frame_available():
        continue

    gettargetposition()

    input("Press Enter to start...")
    timeout_cnt = 0

    with open('debug_data.csv', mode='w') as csv_file:
        fieldnames = ['time', 'x_bb', 'y_bb', 'w_bb', 'h_bb', 'f_u', 'f_v', 'f_delta', 'delta_f_u_psi', 'delta_f_u_y',
                      'delta_f_v_z', 'delta_f_delta_x', 'delta_x_tme', 'delta_y_tme', 'delta_psi_tme', 'delta_z_tme',
                      'prev_delta_x_tme', 'prev_delta_y_tme', 'prev_delta_psi_tme', 'prev_delta_z_tme',
                      'v_xr', 'v_yr', 'v_zr', 'yawrate', 'pitch_angle', 'roll_angle', 'yaw_angle']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            await asyncio.sleep(controller_period)
            csv_row['time'] = time.time()
            async for euler_angles in drone.telemetry.attitude_euler():
                csv_row['pitch_angle'] = euler_angles.pitch_deg
                csv_row['roll_angle'] = euler_angles.roll_deg
                csv_row['yaw_angle'] = euler_angles.yaw_deg
                break

            v_xr, v_yr, yawrate, v_zr = getsetpoints()

            if v_xr is None:
                timeout_cnt = timeout_cnt + controller_period
                if timeout_cnt > 1/5:
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0, 0, 0, 0)
                    )
                continue

            # print("vxr: " + str(v_xr) + " vyr: " + str(v_yr) + " vzr: " + str(v_zr) + " yawrater: " + str(yawrate))
            # async for odometry in drone.telemetry.odometry():
            #     print(odometry)
            #     break

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(v_xr, v_yr, v_zr, yawrate)
            )
            writer.writerow(csv_row)
            timeout_cnt = 0


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
