import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def point_vis(point, fig, ax,  point_type="None"):
    x = point[:, 0]
    y = point[:, 1]
    z = point[:, 2]

    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    z_max = np.max(z)
    z_min = np.min(z)

    # fig = plt.figure()
    # ax = Axes3D(fig)

    ax.scatter(x, y, z, c='r', marker='^')
    for i in range(x.shape[0]):
        ax.text(x[i], y[i], z[i], i)

    ind_start = []
    ind_end = []
    if point_type == "landmark":
        ind_start = [5, 20, 6, 7, 20, 8, 9, 10, 20, 11, 12, 13, 20, 14, 15, 16, 20, 17, 18, 19]
        ind_end = [20, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
    elif point_type == "joint":
        ind_start = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
        ind_end = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21]
    elif point_type == "None":
        ind_start = []
        ind_end = []
    else:
        print("没有这种类型的点")
        return

    if point_type != "None":
        x_line = []
        y_line = []
        z_line = []
        for i in range(len(ind_start)):
            start = ind_start[i]
            end = ind_end[i]
            x_line = [x[start], x[end]]
            y_line = [y[start], y[end]]
            z_line = [z[start], z[end]]
            ax.plot(x_line, y_line, z_line, 'b')


    ax.set_xlabel('X label')  # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    ax.set_xlim3d(xmin=x_min, xmax=x_min + 200)
    ax.set_ylim3d(ymin=y_min, ymax=y_min + 200)
    ax.set_zlim3d(zmin=z_min, zmax=z_min + 200)

    plt.show()

def point_vis_two_hands(point_two_hands, fig, point_type="None"):

    left_hand = point_two_hands[0]
    right_hand = point_two_hands[1]

    ax = Axes3D(fig)

    point_vis(left_hand, fig, ax, point_type=point_type)
    point_vis(right_hand, fig, ax, point_type=point_type)

def draw_landmark_to_img(image, landmark2d):
    '''画单手'''
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # landmark 连接关系
    ind_start = [5, 20, 6, 7, 20, 8, 9, 10, 20, 11, 12, 13, 20, 14, 15, 16, 20, 17, 18, 19]
    ind_end = [20, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]

    x = landmark2d[:, 0]
    y = landmark2d[:, 1]
    for i in range(len(ind_start)):
        start = ind_start[i]
        end = ind_end[i]

        point_start = (int(x[start]), int(y[start]))
        point_end = (int(x[end]), int(y[end]))

        cv2.line(image, point_start, point_end, (0, 255, 0), 2)


    for landmark_idx, landmark in enumerate(landmark2d):
        # point = tuple(int(landmark[0]), int(landmark[1]))
        cv2.circle(image, (int(landmark[0]), int(landmark[1])), 3, (0, 0, 255), -1)

    cv2.imshow("Image with Points", image)
    cv2.waitKey(1)

def draw_landmark_to_img_two_hands(image, landmark2d, draw_cam_id):
    '''画双手'''
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    ind_start = [5, 20, 6, 7, 20, 8, 9, 10, 20, 11, 12, 13, 20, 14, 15, 16, 20, 17, 18, 19]
    ind_end = [20, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]

    for hand_idx in landmark2d.keys():
        if draw_cam_id not in landmark2d[hand_idx].keys():
            continue
        x = landmark2d[hand_idx][draw_cam_id][:, 0]
        y = landmark2d[hand_idx][draw_cam_id][:, 1]

        for i in range(len(ind_start)):
            start = ind_start[i]
            end = ind_end[i]
            point_start = (int(x[start]), int(y[start]))
            point_end = (int(x[end]), int(y[end]))
            if hand_idx == 0:
                color = (0, 255, 0)
            elif hand_idx == 1:
                color = (255, 0, 0)
            cv2.line(image, point_start, point_end, color, 2)

        for landmark_idx, landmark in enumerate(landmark2d[hand_idx][draw_cam_id]):
            # point = tuple(int(landmark[0]), int(landmark[1]))
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)


    cv2.imshow("Image with Points", image)
    cv2.waitKey(1)


def save_draw_landmark_to_img_two_hands(image, landmark2d, draw_cam_id):
    '''画双手'''
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img_origin = image.copy()
    ind_start = [5, 20, 6, 7, 20, 8, 9, 10, 20, 11, 12, 13, 20, 14, 15, 16, 20, 17, 18, 19]
    ind_end = [20, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]

    for hand_idx in landmark2d.keys():
        if draw_cam_id not in landmark2d[hand_idx].keys():
            continue
        x = landmark2d[hand_idx][draw_cam_id][:, 0]
        y = landmark2d[hand_idx][draw_cam_id][:, 1]

        for i in range(len(ind_start)):
            start = ind_start[i]
            end = ind_end[i]
            point_start = (int(x[start]), int(y[start]))
            point_end = (int(x[end]), int(y[end]))
            if hand_idx == 0:
                color = (0, 255, 0)
            elif hand_idx == 1:
                color = (255, 0, 0)
            cv2.line(image, point_start, point_end, color, 2)

        for landmark_idx, landmark in enumerate(landmark2d[hand_idx][draw_cam_id]):
            # point = tuple(int(landmark[0]), int(landmark[1]))
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)

    return image # image


def draw_detection_box(image, center_length, draw_cam_id):
    '''
    画手部检测框 正方形
    input: image  框中心+边长  相机id
    '''
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for hand_idx in center_length.keys():

        if draw_cam_id not in center_length[hand_idx].keys():
            continue

        center_length_arr = center_length[hand_idx][draw_cam_id]
        center_x = center_length_arr[0]
        center_y = center_length_arr[1]
        length = center_length_arr[2]
        # 定义矩形的起点和终点坐标
        start_point = (int(center_x - length/2) , int(center_y - length/2))
        end_point = (int(center_x + length/2) , int(center_y + length/2))

        if hand_idx == 0:
            color = (0, 255, 0) # 左手 绿色
        elif hand_idx == 1:
            color = (255, 0, 0) # 右手 蓝色

        cv2.rectangle(image, start_point, end_point, color, 2)

    cv2.imshow("Image with box", image)
    cv2.waitKey(1)


def draw_test(image, center, length):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    center_x = center[0]
    center_y = center[1]

    start_point = (int(center_x - length / 2), int(center_y - length / 2))
    end_point = (int(center_x + length / 2), int(center_y + length / 2))

    cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)
    cv2.imshow("Image with box", image)
    cv2.waitKey(1)


def draw_box_pre_frame(image, center_length, cls):
    '''利用标签信息画检测框'''
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for i, hand_conf in enumerate(cls):
        if hand_conf == 1:
            center_x = center_length[i][0]
            center_y = center_length[i][1]
            length = center_length[i][2]
            # 定义矩形的起点和终点坐标
            start_point = (int(center_x - length / 2), int(center_y - length / 2))
            end_point = (int(center_x + length / 2), int(center_y + length / 2))

            if i == 0:
                color = (0, 255, 0)  # 左手 绿色
            elif i == 1:
                color = (255, 0, 0)  # 右手 蓝色

            cv2.rectangle(image, start_point, end_point, color, 2)

    cv2.imshow("Image with box", image)
    cv2.waitKey(100)