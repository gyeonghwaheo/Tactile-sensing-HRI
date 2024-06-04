import os
import telnetlib
from ftplib import FTP
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import datetime
import time
import socket
import pandas as pd


def shot_show(save_filename):
    ip = "192.168.1.2"
    user = 'admin'
    password = ''

    tn = telnetlib.Telnet(ip)
    telnet_user = user + '\r\n'
    tn.write(telnet_user.encode('ascii'))
    tn.write("\r\n".encode('ascii'))

    tn.write(b"SFM0163.5\r\n")

    ftp = FTP(ip)
    ftp.login(user)

    filename = "image" + '.bmp'
    tn.write(b"SE8\r\n")
    time.sleep(0.5)
    tn.write(b"SE8\r\n")

    lf = open(filename, "wb")
    ftp.retrbinary("RETR " + filename, lf.write)
    lf.close()

    image = cv2.imread(filename)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("image", image)
    cv2.waitKey(500)
    cv2.imwrite(save_filename, image)

    cv2.waitKey(500)
    cv2.destroyWindow("image")


def find_midpoint(points):
    x_coords = [x for x, y in points]
    y_coords = [y for x, y in points]
    midpoint = (sum(x_coords) / len(points), sum(y_coords) / len(points))
    return midpoint


def translate_points(points, translation):
    return [(x + translation[0], y + translation[1]) for x, y in points]


def rotate_points(points, angle_degrees, rotation_center):
    angle_rad = math.radians(angle_degrees)

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    translated_points = translate_points(points, (-rotation_center[0], -rotation_center[1]))

    points_matrix = np.array(translated_points)
    rotated_matrix = np.dot(points_matrix, rotation_matrix)

    rotated_translated_points = translate_points(rotated_matrix, rotation_center)
    rotated_points = [tuple(point) for point in rotated_translated_points]

    return rotated_points


def apply_displacement(points, x_displacement, y_displacement):
    displaced_points = [(x + x_displacement, y + y_displacement) for x, y in points]
    return displaced_points


def cognex_img_processing(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    crop_image = blurred[240:720, 320:960]

    edges = cv2.Canny(crop_image, 50, 150, apertureSize=3)


    edge_points = np.argwhere(edges != 0)

    if edge_points.size == 0:
        print("No edge points were found")
        return

    max_distance = 0
    point1 = point2 = (0, 0)
    for i in range(len(edge_points)):
        for j in range(i + 1, len(edge_points)):
            pt1 = edge_points[i]
            pt2 = edge_points[j]
            distance = np.linalg.norm(pt1 - pt2)

            if distance > max_distance:
                max_distance = distance
                point1 = pt1
                point2 = pt2

    point1 = (point1[1], point1[0])
    point2 = (point2[1], point2[0])

    print(f"Line between points: {point1} to {point2}")

    cv2.line(image, point1, point2, (20, 220, 20), 2)

    midpoint_new = find_midpoint([point1, point2])
    print(midpoint_new)
    angle_new = (point1[1] - point2[1]) / (point1[0] - point2[0])
    print(angle_new)
    print(midpoint_new[0])
    cv2.circle(image, (int(midpoint_new[0]), int(midpoint_new[1])), 5, (0, 0, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_start = f'{point1}'
    text_end = f'{point2}'
    cv2.putText(image, text_start, point1, font, 0.5, (20, 220, 20), 1, cv2.LINE_AA)
    cv2.putText(image, text_end, point2, font, 0.5, (20, 220, 20), 1, cv2.LINE_AA)
    cv2.putText(image, f'{angle_new}', (int(midpoint_new[0]), int(midpoint_new[1])), font, 0.5, (20, 220, 20), 1,
                cv2.LINE_AA)

    cv2.imshow('Image with Line between points', image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    return (midpoint_new[0], midpoint_new[1]), angle_new


def plot_points(before, after, rotation_center):
    x_before, y_before = zip(*before)
    x_after, y_after = zip(*after)

    fig, ax = plt.subplots()

    ax.scatter(x_before, y_before, color='red', label='Before rotation')
    ax.scatter(x_after, y_after, color='blue', label='After rotation')

    for point_before, point_after in zip(before, after):
        ax.plot([point_before[0], point_after[0]], [point_before[1], point_after[1]], 'r--', linewidth=0.5)

    ax.set_aspect('equal', adjustable='box')
    ax.scatter(*rotation_center, color='green', marker='x', label='Center of rotation')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Rotated points')

    ax.legend()

    ax.grid(True)

    plt.show(block=False)
    plt.pause(2)
    plt.close()


def heo_send(client_socket, a):
    client_socket.sendall(a.encode())
    data = client_socket.recv(1024)
    print('Yaskawa :', repr(data.decode()))


def heo_recv(client_socket):
    data = client_socket.recv(1024)
    print('Yaskawa :', repr(data.decode()))
    time.sleep(0.005)


def calculate_distance(current_pos, target_pos):
    return math.sqrt(
        (current_pos[0] - target_pos[0]) ** 2 +
        (current_pos[1] - target_pos[1]) ** 2 +
        (current_pos[2] - target_pos[2]) ** 2)


def wait_until_reached_target(client_socket, target_pos):
    while True:
        heo_send(client_socket, 'HOSTCTRL_REQUEST RPOSC 4\r\n')
        client_socket.sendall('0,0\r'.encode())
        data = client_socket.recv(1024)
        data = data.decode()
        data = data.split(',')
        print(data[0], data[1], data[2])

        current_pos = (float(data[0]), float(data[1]), float(data[2]))
        distance = calculate_distance(current_pos, target_pos)

        if distance <= 1:
            break

        time.sleep(1)


def get_state(client_socket):
    heo_send(client_socket, 'HOSTCTRL_REQUEST RSTATS 0\r\n')
    yrc_1000_state = client_socket.recv(1024)

    data_list = yrc_1000_state.decode().split(',')
    number_string = data_list[0]
    number = int(number_string)
    run_bit = bin(number)[6]

    print('Yaskawa :', run_bit)
    time.sleep(0.01)


original_points = ["List of coordinate values for the reference incision line"]

midpoint_ori = find_midpoint(original_points)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST = 'IP address'
PORT = 80
client_socket.connect((HOST, PORT))
heo_send(client_socket, 'CONNECT Robot_access Keep-Alive:9000\r\n')

get_state(client_socket)

heo_send(client_socket, 'HOSTCTRL_REQUEST SVON 2\r\n')
heo_send(client_socket, '1\r')

x_z_angle_speed_info_1 = [[000, 000, 000, 000]]


for i in x_z_angle_speed_info_1:
    print('Label :', i)

    x_value, z_value, ang_value, sp_value = i
    target_coordinate = f'0,{sp_value:.1f},0,{x_value:.3f},0.000,{z_value:.3f},{ang_value:.4f},-0.0000,90.0000,1,0,0,0,0,0,0,0\r'
    bytes_length = len(target_coordinate.encode('utf-8'))

    heo_send(client_socket, f'HOSTCTRL_REQUEST MOVL {bytes_length}\r\n')
    heo_send(client_socket, target_coordinate)

    wait_until_reached_target(client_socket, [x_value, 0, z_value])

today = datetime.date.today()

year = today.year
month = today.month
day = today.day

today_year_month_day_number = f"{year:04d}{month:02d}{day:02d}"[:8]

if not os.path.exists('img_cog/' + today_year_month_day_number):
    os.mkdir('img_cog/' + today_year_month_day_number)

now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
save_filename = 'img_cog/' + today_year_month_day_number + f'/{now}.png'

shot_show(save_filename)

midpoint_new, angle_new = cognex_img_processing(save_filename)
print(midpoint_new, angle_new)

radians = math.atan(angle_new)
angle = math.degrees(radians)

x_displacement = (10 / 72) * (midpoint_new[0] - 000)
y_displacement = (10 / 72) * (midpoint_new[1] - 000)

rotated_points = rotate_points(original_points, angle, midpoint_ori)
new_points = apply_displacement(rotated_points, x_displacement, y_displacement)

plot_points(original_points, new_points, midpoint_ori)

x_y_info = new_points

x_value, y_value = x_y_info[0]
z_value = 000

angle_pre = 90-angle
target_coordinate = f'0,40.0,0,{x_value:.3f},{y_value:.3f},{z_value:.3f},-180.0000,0.0000,{angle_pre:.4f},1,0,0,0,0,0,0,0\r'
bytes_length = len(target_coordinate.encode('utf-8'))

heo_send(client_socket, f'HOSTCTRL_REQUEST MOVL {bytes_length}\r\n')
heo_send(client_socket, target_coordinate)

wait_until_reached_target(client_socket, [x_value, y_value, z_value])

angle_info = ["List of predicted angle values for the robotic surgery"]

x_y_for_imove = [(x_y_info[i+1][0] - x_y_info[i][0], x_y_info[i+1][1] - x_y_info[i][1]) for i in range(len(x_y_info)-1)]
x_y_angle_info = [list(x) + [y] for x, y in zip(x_y_for_imove, angle_info)]
print(x_y_angle_info)

h_num = 0
for i in x_y_angle_info:
    print('Label :', i)
    x_imove, y_imove, angle_imove = i

    if angle_imove != 0:
        target_coordinate = f'1,5.0,1,0.000,0.000,0.000,0.0000,{-angle_imove:.4f},0.0000,0,0,0,0,0,0,0,0\r'
        bytes_length = len(target_coordinate.encode('utf-8'))

    heo_send(client_socket, f'HOSTCTRL_REQUEST IMOV {bytes_length}\r\n')
    heo_send(client_socket, target_coordinate)
    time.sleep(0.5)

    target_coordinate = f'1,1.0,0,{x_imove:.3f},{y_imove:.3f},0.000,0.0000,0.0000,0.0000,0,0,0,0,0,0,0,0\r'
    bytes_length = len(target_coordinate.encode('utf-8'))

    heo_send(client_socket, f'HOSTCTRL_REQUEST IMOV {bytes_length}\r\n')
    heo_send(client_socket, target_coordinate)

    time.sleep(0.5)


x_z_speed_info_2 = ["List of coordinate values"]

for i in x_z_speed_info_2:
    print('Label :', i)

    x_value, z_value, sp_value = i
    target_coordinate = f'0,{sp_value:.1f},0,{x_value:.3f},0.000,{z_value:.3f},-180.0,-0.0000,90.0000,1,0,0,0,0,0,0,0\r'
    bytes_length = len(target_coordinate.encode('utf-8'))

    heo_send(client_socket, f'HOSTCTRL_REQUEST MOVL {bytes_length}\r\n')
    heo_send(client_socket, target_coordinate)

    wait_until_reached_target(client_socket, [x_value, 0, z_value])

time.sleep(1)

heo_send(client_socket, 'HOSTCTRL_REQUEST SVON 2\r\n')
heo_send(client_socket, '0\r')
time.sleep(1.0)

client_socket.close()
