import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch.nn as nn
import torch
import cv2
import time

class GestureRecognition():
    def __init__(self, SH_ges_rec_model, HH_ges_rec_model, frame_gap, window_size, label_name=[]):
        self._device: str = "cuda" if torch.cuda.device_count() else "cpu"

        self.frame_gap = frame_gap
        self.window_size = window_size
        self.label_name = label_name

        self._SH_ges_rec_model = SH_ges_rec_model
        self._SH_ges_rec_model.to(self._device) # 单手模型
        self._HH_ges_rec_model = HH_ges_rec_model
        self._HH_ges_rec_model.to(self._device) # 单手模型



        self.gap_num = 0
        #self.gesture_windows = []
        self.gesture_list_128 = []

    def seq_translation(self, skes_joints):
        """
        skes_joints: T V*C
        计算所有帧关于第一帧根节点（序号0）的相对坐标
        """
        joint_num = int(skes_joints.shape[-1] / 3)  # 21
        num_frames = skes_joints.shape[0]
        num_bodies = 1
        i = 0  # get the "real" first frame of actor1
        origin = np.copy(skes_joints[i, 0:3])  # new origin: joint-0
        for f in range(num_frames):
            #test = np.tile(origin, 59)
            skes_joints[f] -= np.tile(origin, joint_num)

        return skes_joints

    def preprocessing(self, input):
        # print(input.shape)
        T, V, C = input.shape
        input = input.reshape(T, -1)  # T V*C
        input = self.seq_translation(input)  # 相对于序列第一帧的坐标
        input = np.expand_dims(input, axis=0)
        # print(input.shape)
        return input

    def SH_REC(self, input):
        """单手手势识别"""
        input = input[:, 1, :, :] # 只取右手
        input = self.preprocessing(input)
        N, T, _ = input.shape
        input = input.reshape((N, T, 1, 21, 3)).transpose(0, 4, 1, 3, 2)

        input = torch.from_numpy(input).to(self._device)
        output = self._SH_ges_rec_model(input).data.cpu().numpy()

        # _, predict_label = torch.max(output.data, 1)
        output_softmax = self.softmax(output)
        predict_label = int(np.argmax(output, 1))
        prob = float(np.max(output_softmax, 1))
        prob = round(prob, 2)

        # start_time = time.time()
        # end_time = time.time()
        # gaptime = end_time - start_time  # 计算经过的时间
        # print(gaptime * 1000, 'ms')

        return predict_label, prob, output_softmax

    def HH_REC(self, input):
        """双手手势识别"""
        input = input.reshape(input.shape[0], -1, 3)
        input = self.preprocessing(input)
        N, T, _ = input.shape

        # # 左手镜像为右手
        # input = input.reshape((N, T, 2, 21, 3))
        # data_l = input[:,:,0,:,:]
        # data_r = input[:,:,1,:,:]
        # data_l2r = np.zeros_like(data_l)
        # data_l2r[:, :, :, 0] = data_l[:, :, :, 0]
        # data_l2r[:, :, :, 1] = -data_l[:, :, :, 1]
        # data_l2r[:, :, :, 2] = data_l[:, :, :, 2]
        # input = np.concatenate([data_l2r, data_r], axis=2).reshape((N, T, 2, 21, 3)).transpose(0, 4, 1, 3, 2)

        input = input.reshape((N, T, 1, 42, 3)).transpose(0, 4, 1, 3, 2)

        input = torch.from_numpy(input).to(self._device)
        output = self._HH_ges_rec_model(input).data.cpu().numpy()
        # _, predict_label = torch.max(output.data, 1)
        output_softmax = self.softmax(output)
        predict_label = int(np.argmax(output, 1)) + 17 # 双手动作从17开始
        prob = float(np.max(output_softmax, 1))
        prob = round(prob, 2)


        return predict_label, prob, output_softmax

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        # test = e_x.sum(axis=1)
        result = e_x / e_x.sum(axis=1)

        return result

    def Recognition(self, joint_posi, valid_tracking):

        joint_posi_cp = joint_posi.copy()

        # print(joint_posi_cp[0][0])
        predict_label = None
        gesture_windows = []
        prob = 0.0
        output_softmax = []

        if len(self.gesture_list_128) == 64:  # 这里的128为你想要固定长度list的长度
            self.gesture_list_128.pop(0)  # 删除list a 中的第一个元素
            self.gesture_list_128.append(joint_posi_cp)  # 添加list a 中的最后一个元素
        else:
            self.gesture_list_128.append(joint_posi_cp)

        #print(len(self.gesture_list_128))

        if self.gap_num == self.frame_gap:
            # gap_num = 0
            gesture_windows = self.gesture_list_128[-self.window_size:]  # 将帧传入list 到24帧后停止
        #print(len(gesture_windows))

        if len(gesture_windows) == 0:
            self.gap_num = self.gap_num + 1

        #print('gesture_windows:', len(gesture_windows), 'gap_num:', self.gap_num)

        if len(gesture_windows) == self.window_size:  # 24帧
            input = np.array(gesture_windows, dtype=np.float32)


            # if valid_tracking[0] == True and valid_tracking[1] == True: # 双手
            #     # print('双手')
            #     predict_label, prob, output_softmax = self.HH_REC(input)
            # elif valid_tracking[0] == False and valid_tracking[1] == True: # 单右手
            #     # print('单右手')
            #     predict_label, prob, output_softmax = self.SH_REC(input)
            #     if predict_label>=17:
            #         predict_label = predict_label + 4
            # elif valid_tracking[0] == True and valid_tracking[1] == False: # 单左手
            #     # print('单左手')
            #     pass
            # else:
            #     # print('无手')
            #     pass
            
            if valid_tracking[1] == True: # 单右手
                # print('单右手')
                predict_label, prob, output_softmax = self.SH_REC(input)
                if predict_label>=17:
                    predict_label = predict_label + 4
            else:
                pass

            # start_time = time.time()
            # end_time = time.time()
            # gaptime = end_time - start_time  # 计算经过的时间
            # print(gaptime * 1000, 'ms')

            gesture_windows = []
            self.gap_num = 0

        return predict_label, prob, output_softmax