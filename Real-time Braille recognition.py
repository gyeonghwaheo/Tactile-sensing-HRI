import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time
import pyvisa
import pandas as pd
import datetime
import os
import numpy as np
import socket
import multiprocessing
import random
import turtle
import tensorflow as tf


def connect_to_device(device_address):
    inst = pyvisa.ResourceManager().open_resource(device_address)

    inst.timeout = 1000
    inst.write("RATE M")
    inst.read()
    time.sleep(1)

    return inst


def get_measurement_data(inst):
    val = inst.read()

    try:
        v_num = float(val)
        Res_val = (5000 * v_num) / (5 - v_num)
        Res_val = round(Res_val, 3)
    except ValueError:
        print("Error")
        Res_val = 0

    time.sleep(0.001)

    return Res_val


def heo_send(client_socket, a):
    client_socket.sendall(a.encode())
    data = client_socket.recv(1024)
    print('Yaskawa :', repr(data.decode()))


def heo_recv(client_socket):
    data = client_socket.recv(1024)
    print('Yaskawa :', repr(data.decode()))
    time.sleep(0.005)


def get_pos(client_socket, start_time, yrc_1000_shared_data, label_num):
    heo_send(client_socket, 'HOSTCTRL_REQUEST RPOSC 4\r\n')
    client_socket.sendall('0,0\r'.encode())
    data = client_socket.recv(1024)
    data = data.decode()
    data = data.split(',')
    print(data[0], data[1], data[2])

    yrc_1000_shared_data['time_data'].append(time.time() - start_time)
    yrc_1000_shared_data['x_data'].append(data[0])
    yrc_1000_shared_data['y_data'].append(data[1])
    yrc_1000_shared_data['z_data'].append(data[2])
    yrc_1000_shared_data['Braille'].append(label_num)
    time.sleep(0.005)


def get_state(client_socket):
    heo_send(client_socket, 'HOSTCTRL_REQUEST RSTATS 0\r\n')
    yrc_1000_state = client_socket.recv(1024)

    data_list = yrc_1000_state.decode().split(',')
    number_string = data_list[0]
    number = int(number_string)
    run_bit = bin(number)[7]

    print('Yaskawa :', run_bit)
    time.sleep(0.005)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def min_max_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)
    return normalized


def multimeter_process(multimeter_shared_data, terminate_flag, device_address, start_time):
    fluke = connect_to_device(device_address)
    while terminate_flag.value == 0:
        R_data = get_measurement_data(fluke)
        time_from_0 = time.time() - start_time

        multimeter_shared_data['time_data'].append(time_from_0)
        multimeter_shared_data['resistance_data'].append(R_data)

        time.sleep(0.001)


def plot_update_process(lcr_meter_shared_data, terminate_flag):
    fig, ax1 = plt.subplots(figsize=(16, 4))
    fig.suptitle('Real-time resistance')
    ax1.set_ylabel('Resistance (ohms)')
    ax1.set_xlabel('Time (s)')

    if plt.get_backend() == "TkAgg":
        canvas = FigureCanvasTkAgg(fig, master=fig.canvas.get_tk_widget().master)
        canvas.get_tk_widget().master.geometry('+150+100')

    line1, = ax1.plot([], [], 'b-')
    plt.show(block=False)

    while terminate_flag.value == 0:
        if len(lcr_meter_shared_data['time_data']) >= 200:
            time_data = lcr_meter_shared_data['time_data'][-200:]
        else:
            time_data = lcr_meter_shared_data['time_data']

        if len(lcr_meter_shared_data['resistance_data']) >= 200:
            resistance_data = lcr_meter_shared_data['resistance_data'][-200:]
        else:
            resistance_data = lcr_meter_shared_data['resistance_data']

        min_length = min(len(time_data), len(resistance_data))

        time_data = time_data[:min_length]
        resistance_data = resistance_data[:min_length]

        if time_data and resistance_data:
            line1.set_xdata(time_data)
            line1.set_ydata(resistance_data)
            ax1.relim()
            ax1.autoscale_view()
            plt.draw()
            plt.pause(0.001)


def yrc_1000_process(st_time, yrc_1000_shared_data, x_y_z_coord, multimeter_shared_data):
    loaded_model = tf.keras.models.load_model('Trained_model', compile=False)
    Braille_labels = ['e', 'l', 'c', 't', 'r', 'o', 'n', 'i', 's', 'k']

    win = turtle.Screen()
    win.setup(width=1500, height=300, startx=200, starty=600)
    t = turtle.Turtle()
    t.penup()
    t.hideturtle()
    t.goto(-600, 0)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    HOST = '192.168.000.000'
    PORT = 80
    client_socket.connect((HOST, PORT))

    heo_send(client_socket, 'CONNECT Robot_access Keep-Alive:9000\r\n')
    get_state(client_socket)
    get_pos(client_socket, st_time, yrc_1000_shared_data, 0)
    heo_send(client_socket, 'HOSTCTRL_REQUEST SVON 2\r\n')
    heo_send(client_socket, '1\r')


    x_value, y_value, z_value = x_y_z_coord[0]
    target_coordinate = f'0,50.0,0,{x_value:.3f},{y_value:.3f},{z_value:.3f},-180.0000,-0.0000,90.0000,1,0,0,0,0,0,0,0\r'
    bytes_length = len(target_coordinate.encode('utf-8'))

    heo_send(client_socket, f'HOSTCTRL_REQUEST MOVL {bytes_length}\r\n')
    heo_send(client_socket, target_coordinate)
    time.sleep(7)
    get_pos(client_socket, st_time, yrc_1000_shared_data, 0)

    for loop_num in range(1):
        print(loop_num)
        for i in range(1, 18):
            get_pos(client_socket, st_time, yrc_1000_shared_data, i)

            if i in {17}:
                x_value, y_value, z_value = x_y_z_coord[i]
                target_coordinate = f'0,50.0,0,{x_value:.3f},{y_value:.3f},{z_value:.3f},-180.0000,-0.0000,90.0000,1,0,0,0,0,0,0,0\r'
                bytes_length = len(target_coordinate.encode('utf-8'))

                heo_send(client_socket, f'HOSTCTRL_REQUEST MOVL {bytes_length}\r\n')
                heo_send(client_socket, target_coordinate)
                time.sleep(3)
            else:
                x_value, y_value, z_value = x_y_z_coord[i]
                target_coordinate = f'0,10.0,0,{x_value:.3f},{y_value:.3f},{z_value:.3f},-180.0000,-0.0000,90.0000,1,0,0,0,0,0,0,0\r'
                bytes_length = len(target_coordinate.encode('utf-8'))

                heo_send(client_socket, f'HOSTCTRL_REQUEST MOVL {bytes_length}\r\n')
                heo_send(client_socket, target_coordinate)
                for_predict_point = len(list(multimeter_shared_data['resistance_data']))
                print(for_predict_point)
                time.sleep(4)

                if i > 1 and i < 16:
                    for_predict_data = np.array(
                        list(multimeter_shared_data['resistance_data'])[for_predict_point - 10:for_predict_point + 60])
                    print(for_predict_data)

                    for_predict_data = np.array(min_max_normalize(for_predict_data))
                    print(for_predict_data)

                    for_predict_data = for_predict_data.reshape(1, "Length of window", 1)
                    Braille_pre_raw = loaded_model.predict(for_predict_data)
                    Braille_pre = np.argmax(Braille_pre_raw, axis=1)
                    print(Braille_pre)

                    t.write(Braille_labels[Braille_pre[0]], align="center", font=("Arial", 70, "bold"))
                    t.forward(89)

            time.sleep(0.1)

    time.sleep(5)
    get_pos(client_socket, st_time, yrc_1000_shared_data, 0)

    heo_send(client_socket, 'HOSTCTRL_REQUEST SVON 2\r\n')
    heo_send(client_socket, '0\r')
    time.sleep(1.0)

    client_socket.close()


def generate_coordinates():
    start_x, end_x, start_y, start_z = "List of coordinate values"

    float_yaxis = "Offset between Braille sheet and e-skin"
    inter_braille = "Pitch between Braille"

    x_y_z_list = [[start_x, start_y + float_yaxis, start_z], [start_x, start_y, start_z]]

    for _ in range(14):
        new_coord = list(x_y_z_list[-1])
        new_coord[0] += inter_braille
        x_y_z_list.append(new_coord)

    x_y_z_list.append([end_x, start_y + float_yaxis, start_z])
    x_y_z_list.append([start_x, start_y + float_yaxis, start_z])

    return x_y_z_list


if __name__ == "__main__":
    device_address = "Instrument"

    manager = multiprocessing.Manager()
    multimeter_shared_data = manager.dict()
    multimeter_shared_data['time_data'] = manager.list()
    multimeter_shared_data['resistance_data'] = manager.list()

    yrc_1000_shared_data = manager.dict()
    yrc_1000_shared_data['time_data'] = manager.list()
    yrc_1000_shared_data['x_data'] = manager.list()
    yrc_1000_shared_data['y_data'] = manager.list()
    yrc_1000_shared_data['z_data'] = manager.list()
    yrc_1000_shared_data['braille'] = manager.list()

    terminate_flag = manager.Value('i', 0)

    x_y_z_coord = generate_coordinates()
    print('XYZ coordinates :', x_y_z_coord)

    start_time = time.time()

    process_1 = multiprocessing.Process(target=multimeter_process,
                                        args=(multimeter_shared_data, terminate_flag, device_address, start_time))
    process_2 = multiprocessing.Process(target=plot_update_process,
                                        args=(multimeter_shared_data, terminate_flag))
    process_3 = multiprocessing.Process(target=yrc_1000_process,
                                        args=(start_time, yrc_1000_shared_data, x_y_z_coord, multimeter_shared_data))

    process_1.start()
    process_2.start()
    process_3.start()

    time.sleep(5)

    process_3.join()
    terminate_flag.value = 1
    process_2.join()
    process_1.join()
