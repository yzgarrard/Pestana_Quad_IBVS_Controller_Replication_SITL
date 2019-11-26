from pymavlink import mavutil

connection_from_sitl = mavutil.mavlink_connection(device="udp::14540")

while True:
    #recv_msg = con.recv_msg()
    recv_msg = connection_from_sitl.recv_match()
    if not recv_msg:
        continue
    if recv_msg.get_type() == "LOCAL_POSITION_NED":
        print(recv_msg)


