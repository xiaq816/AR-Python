import numpy as np
import json
import re
import csv
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import socket
import threading
import asyncio
import websockets
from datetime import datetime
import queue

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理numpy类型"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class SocketClient:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}/ws/mqtt/"
        self.received_count = 0
        self.data_buffer = []  # 数据缓存列表
        self.buffer_max_size = 100  # 最大缓存数量
        self.running = True
        self.buffer_lock = threading.Lock()  # 线程锁保证数据安全

    def log(self, message):
        """打印带时间戳的日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    async def receive_data(self):
        """接收服务器数据"""
        self.log(f"🔗 尝试连接到服务器 {self.uri}")

        try:
            async with websockets.connect(self.uri) as websocket:
                self.log("✅ 成功连接到服务器")

                # 发送一条测试消息
                test_message = {
                    "type": "client_ready",
                    "message": "客户端已准备接收机械臂数据",
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(test_message, ensure_ascii=False))
                self.log("📤 发送客户端就绪消息")

                while self.running:
                    try:
                        # 接收数据
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        try:
                            # 尝试解析JSON数据
                            data = json.loads(message)
                            message_content = data.get('message')
                            if message_content:
                                data_content = json.loads(message_content)
                                command_type = data_content.get('command_type')
                                if command_type == "1":
                                    self.received_count += 1

                                    # 解析关节数据
                                    joint_values = [float(x) for x in data_content.get('data').split(",")]

                                    # 将数据放入缓存
                                    record = {
                                        'timestamp': data_content.get('timestamp'),
                                        'client_timestamp': data_content.get('client_timestamp'),
                                        'command_type': command_type,
                                        'joint_angles_deg': joint_values,
                                        'joint_angles_rad': np.radians(joint_values)
                                    }

                                    with self.buffer_lock:
                                        # 如果缓存已满，移除最旧的数据
                                        if len(self.data_buffer) >= self.buffer_max_size:
                                            self.data_buffer.pop(0)
                                        self.data_buffer.append(record)

                                    # 实时显示关节数据（简洁格式）
                                    if self.received_count % 1 == 0:  # 每1条显示一次
                                        self.log(
                                            f"🦾 关节角度: [{joint_values[0]:.1f}, {joint_values[1]:.1f}, {joint_values[2]:.1f}, ...]")

                                    # 每收到1条数据打印一次统计信息
                                    if self.received_count % 1 == 0:
                                        self.log(f"📊 已接收 {self.received_count} 条机械臂数据")

                                elif data_content.get("type") == "ack":
                                    # 这是确认消息
                                    self.log(f"✅ 服务器确认: {data_content.get('message', '')}")

                        except json.JSONDecodeError:
                            # 如果不是JSON，直接显示文本
                            self.log(f"📨 收到文本消息: {message}")
                        except Exception as e:
                            self.log(f"❌ 解析数据时出错: {e}")

                    except asyncio.TimeoutError:
                        # 超时，继续等待
                        continue

        except websockets.exceptions.ConnectionClosed:
            self.log("❌ 连接被服务器关闭")
        except websockets.exceptions.InvalidURI:
            self.log(f"❌ 无效的URI: {self.uri}")
        except Exception as e:
            self.log(f"❌ 连接错误: {e}")

    def get_latest_data(self):
        """获取缓存中最新的数据"""
        with self.buffer_lock:
            if self.data_buffer:
                return self.data_buffer[-1]
            return None

    def stop(self):
        self.running = False


# ------------------ Unity TCP 客户端（带重连 + 发送队列） ------------------
class UnityTcpClient:
    def __init__(self, host='192.168.43.2', port=3333, reconnect_interval=1.0, name="UnityTcpClient"):
        self.host = host
        self.port = int(port)
        self.reconnect_interval = reconnect_interval
        self.send_queue = queue.Queue()
        self._sock = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.name = name

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[{self.name}] 启动客户端线程，目标 Unity {self.host}:{self.port}")

    def _run(self):
        while self._running:
            try:
                # 建立连接
                print(f"[{self.name}] 尝试连接 Unity {self.host}:{self.port} ...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                try:
                    sock.connect((self.host, self.port))
                except Exception as e:
                    sock.close()
                    print(f"[{self.name}] 连接失败: {e}，{self.reconnect_interval}s 后重试")
                    time.sleep(self.reconnect_interval)
                    continue

                # 连接成功
                sock.settimeout(None)  # 设为阻塞模式
                with self._lock:
                    self._sock = sock
                print(f"[{self.name}] ✅ 已连接到 Unity {self.host}:{self.port}")

                # 发送循环：从队列取数据并发送
                while self._running:
                    try:
                        # 等待数据（1s超时以便检查 _running 标志）
                        data = self.send_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    # 序列化并发送
                    try:
                        json_data = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)
                        message = (json_data + '\n').encode('utf-8')
                        with self._lock:
                            if self._sock:
                                self._sock.sendall(message)
                    except Exception as e:
                        print(f"[{self.name}] 发送失败: {e}，将尝试重连并重新入队该数据")
                        # 发生发送错误：把数据重新放回队列前端（简单策略：放回队列尾部）
                        try:
                            self.send_queue.put_nowait(data)
                        except:
                            pass
                        # 关闭当前 socket 并退出发送循环以触发重连
                        with self._lock:
                            try:
                                self._sock.close()
                            except:
                                pass
                            self._sock = None
                        break  # 跳出发送循环，回到重连逻辑

                # 退出时关闭 socket
                with self._lock:
                    if self._sock:
                        try:
                            self._sock.close()
                        except:
                            pass
                        self._sock = None

            except Exception as e:
                print(f"[{self.name}] 运行线程捕获异常: {e}")
                time.sleep(self.reconnect_interval)

        # 线程退出：确保 socket 关闭
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except:
                    pass
                self._sock = None
        print(f"[{self.name}] 客户端线程已停止")

    def send(self, data):
        """
        把消息放入发送队列（线程安全）
        data: 可 JSON 序列化的对象（支持 numpy via NumpyEncoder）
        """
        try:
            self.send_queue.put_nowait(data)
            return True
        except Exception as e:
            print(f"[{self.name}] 入队失败: {e}")
            return False

    def stop(self):
        """停止线程并关闭 socket"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except:
                    pass
                self._sock = None
        print(f"[{self.name}] 已停止")

class YueJiangRobot:
    def __init__(self, dh_params=None, base_position=[0, 0, 0], base_orientation=[0, 0, 0]):
        """
        初始化机械臂（单位：毫米）
        Args:
            dh_params: DH参数列表，每个元素为[a, alpha, d, theta_offset]
            base_position: 基座位置 [x, y, z] (世界坐标系原点，单位：毫米)
            base_orientation: 基座姿态 [roll, pitch, yaw] (弧度)
        """

        # 设置机械臂的DH参数
        if dh_params is None:
            self.dh_params = [
                [0.0, np.pi / 2, 230.0, 0],  # 关节1: a=0.0, α=π/2, d=230mm, θ_offset=0
                [-825.2, 0.0, 0.0, -np.pi / 2],  # 关节2: a=-825.2mm, α=0.0, d=0, θ_offset=-π/2
                [-746.0, 0.0, 0.0, 0],  # 关节3: a=-746.0mm, α=0.0, d=0, θ_offset=0
                [0.0, np.pi / 2, 175.6, -np.pi / 2],  # 关节4: a=0.0, α=π/2, d=175.6mm, θ_offset=-π/2
                [0.0, -np.pi / 2, 128.8, 0],  # 关节5: a=0.0, α=-π/2, d=128.8mm, θ_offset=0
                [0.0, 0.0, 136.5, 0]  # 关节6: a=0.0, α=0.0, d=136.5mm, θ_offset=0
            ]
        else:
            self.dh_params = dh_params

        # 存储基座变换
        self.base_position = np.array(base_position)
        self.base_orientation = np.array(base_orientation)

        # 计算基座变换矩阵
        self.base_transform = self.calculate_base_transform()

        # 存储轨迹数据
        self.trajectory = []

        # 舱体中心线参数（根据实际应用场景调整）
        self.cabin_center = np.array([915, 0, 220])  # 舱体中心坐标
        self.cabin_centerline_length = 400  # 舱体中心线长度

        # 定位孔坐标（根据实际应用场景调整）
        self.positioning_hole = np.array([916.8, -286.8, 368.9])  # 定位孔坐标（916.8，-286.8，368.9）
        self.positioning_line_end = np.array([915, -286.8, 220])  # 定位线终点坐标（915， -286.8，220）

        # 机械臂末端光线参数
        self.end_effector_ray_length = 100  # 末端光线长度

        # 一体化设备上的定位孔位置（相对于基底坐标）
        self.positioning_hole_local = np.array([402.7, 237.2, 0])  # 单位：毫米402.7,237.2,0)

        # 设置误差阈值（3mm）
        self.error_threshold = 3.0

        # 存储最近计算值的属性
        self.last_position = None
        self.last_hole_world = None
        self.last_alignment_comp_angles = None
        self.last_perpendicularity_comp_angles = None
        self.last_vertical_axis_error = None

    def calculate_base_transform(self):
        """计算基座变换矩阵 - 根据SDH方法建立坐标系"""
        # 根据SDH方法，基座坐标系与世界坐标系对齐
        roll, pitch, yaw = self.base_orientation

        # 计算旋转矩阵（标准右手坐标系）
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # 标准右手坐标系旋转矩阵
        rotation_matrix = Rz @ Ry @ Rx

        # 创建齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = self.base_position

        return T

    def dh_matrix(self, a, alpha, d, theta):
        """
        计算SDH参数对应的齐次变换矩阵
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        # SDH变换矩阵公式
        T = np.array([
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1]
        ])
        return T

    def forward_kinematics(self, joint_angles):
        """
        正向运动学计算 - 使用SDH方法
        T_6^0 = T_1^0 * T_2^1 * T_3^2 * T_4^3 * T_5^4 * T_6^5
        """
        T = np.eye(4)  # 从单位矩阵开始

        # 应用基座变换
        T = T @ self.base_transform

        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_matrix(a, alpha, d, theta)
            T = T @ T_i  # 矩阵连乘

        return T

    def get_end_effector_pose(self, joint_angles):
        """
        获取末端执行器的位置和姿态
        """
        T = self.forward_kinematics(joint_angles)

        # 提取位置
        position = T[:3, 3]

        # 提取旋转矩阵
        rotation_matrix = T[:3, :3]

        # 转换为欧拉角 (ZYX顺序)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0

        euler_angles = np.array([roll, pitch, yaw])

        return position, rotation_matrix, euler_angles

    def get_end_effector_ray(self, joint_angles):
        """
        获取末端执行器的光线

        Returns:
            start_point: 光线起点 [x, y, z]
            end_point: 光线终点 [x, y, z]
            direction: 光线方向向量
        """
        # 获取末端执行器位姿
        position, rotation_matrix, _ = self.get_end_effector_pose(joint_angles)

        # 光线方向为末端执行器的Z轴方向
        direction = rotation_matrix[:, 2]  # Z轴方向向量

        # 计算光线起点和终点
        half_length = self.end_effector_ray_length
        start_point = position - direction * half_length
        end_point = position + direction * half_length

        return start_point, end_point, direction

    def get_cabin_centerline(self):
        """
        获取舱体中心线

        Returns:
            start_point: 中心线起点 [x, y, z]
            end_point: 中心线终点 [x, y, z]
        """
        # 舱体中心线沿Y轴方向
        half_length = self.cabin_centerline_length
        start_point = self.cabin_center - np.array([0, half_length, 0])
        end_point = self.cabin_center + np.array([0, half_length, 0])

        return start_point, end_point

    def get_positioning_line(self):
        """
        获取定位孔连线

        Returns:
            start_point: 定位孔起点 [x, y, z]
            end_point: 定位孔终点 [x, y, z]
        """
        return self.positioning_hole, self.positioning_line_end

    def calculate_alignment_error(self, ray_direction):
        """
        计算同轴度误差（机械臂末端光线与舱体中心线的对齐误差）

        Args:
            ray_direction: 机械臂末端光线的方向向量

        Returns:
            alignment_error: 同轴度误差（毫米）
            angle_error: 角度误差（度）
        """
        # 舱体中心线的方向向量（Y轴方向）
        cabin_direction = np.array([0, 1, 0])

        # 归一化方向向量
        ray_dir_normalized = ray_direction / np.linalg.norm(ray_direction)
        cabin_dir_normalized = cabin_direction / np.linalg.norm(cabin_direction)

        # 计算方向向量之间的夹角（弧度）
        dot_product = np.dot(ray_dir_normalized, cabin_dir_normalized)
        # 防止数值误差导致超出[-1,1]范围
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

        # 将角度误差转换为线性误差（毫米）
        # 使用光线长度作为参考，误差 = 光线长度 * sin(角度)
        alignment_error = self.end_effector_ray_length * np.sin(angle)

        # 转换为角度
        angle_error = np.degrees(angle)

        return alignment_error, angle_error

    def calculate_perpendicularity_error(self, ray_direction):
        """
        计算垂直度误差（电子舱点连线光线与目标件上连线的垂直度误差）

        Args:
            ray_direction: 机械臂末端光线的方向向量

        Returns:
            perpendicularity_error: 垂直度误差（毫米）
            angle_error: 角度误差（度）
        """
        # 电子舱点连线方向（Z轴负方向）
        cabin_line_direction = np.array([0, 0, -1])

        # 归一化方向向量
        ray_dir_normalized = ray_direction / np.linalg.norm(ray_direction)
        cabin_dir_normalized = cabin_line_direction / np.linalg.norm(cabin_line_direction)

        # 计算方向向量之间的夹角（弧度）
        dot_product = np.dot(ray_dir_normalized, cabin_dir_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

        # 计算与90度的偏差
        deviation_angle = np.abs(angle - np.pi / 2)

        # 转换为线性误差
        perpendicularity_error = self.end_effector_ray_length * np.sin(deviation_angle)

        # 转换为角度
        angle_error = np.degrees(deviation_angle)

        return perpendicularity_error, angle_error

    def calculate_vertical_axis_error(self, target_line_vector):
        """
        计算垂直轴偏差（目标件上孔与轴中心连线和电子舱上的连线在同一个平面上的角度偏差）

        Args:
            target_line_vector: 目标件上孔到轴中心的向量（世界坐标系）[x, y, z]

        Returns:
            angle_error: 角度偏差（度）
        """
        # 电子舱连线向量（从定位孔到定位线终点）
        cabin_vector = self.positioning_line_end - self.positioning_hole

        # 计算两个向量的夹角（弧度）
        dot_product = np.dot(cabin_vector, target_line_vector)
        norm_product = np.linalg.norm(cabin_vector) * np.linalg.norm(target_line_vector)

        # 防止除以零
        if norm_product < 1e-6:
            return 0.0

        # 计算夹角的余弦值
        cos_theta = dot_product / norm_product
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # 计算夹角（弧度）并转换为角度
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_displacement_vector(self, ray_direction):
        """
        计算需要向XYZ方向移动多少毫米来修正误差

        Args:
            ray_direction: 机械臂末端光线的方向向量

        Returns:
            displacement_vector: 需要向XYZ方向移动的向量 [dx, dy, dz]（毫米）
        """
        # 理想方向（舱体中心线方向）
        ideal_direction = np.array([0, 1, 0])

        # 当前方向
        current_direction = ray_direction / np.linalg.norm(ray_direction)

        # 计算方向偏差向量
        direction_error = current_direction - ideal_direction

        # 将方向偏差转换为位移向量（毫米）
        displacement_vector = direction_error * self.end_effector_ray_length

        return displacement_vector

    def calculate_alignment_compensation_angles(self, ray_direction):
        """
        计算同轴度补偿角度（目标件的中心线变化到电子舱中心线同方向所需的角度补偿）

        Args:
            ray_direction: 机械臂末端光线的方向向量

        Returns:
            compensation_angles: 需要补偿的角度 [x_angle, y_angle, z_angle]（度）
        """
        # 理想方向（舱体中心线方向）
        ideal_direction = np.array([0, 1, 0])

        # 当前方向
        current_direction = ray_direction / np.linalg.norm(ray_direction)

        # 计算旋转矩阵将当前方向旋转到理想方向
        v = np.cross(current_direction, ideal_direction)
        s = np.linalg.norm(v)
        c = np.dot(current_direction, ideal_direction)

        if s < 1e-6:
            # 方向相同或相反，不需要旋转
            return np.array([0.0, 0.0, 0.0])

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)

        # 从旋转矩阵中提取欧拉角（ZYX顺序）
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        # 转换为角度
        compensation_angles = np.degrees([roll, pitch, yaw])
        return compensation_angles

    def calculate_perpendicularity_compensation_angles(self, ray_direction):
        """
        计算垂直度补偿角度（目标件的孔与轴中心的连线变化到电子舱连线同方向所需的角度补偿）

        Args:
            ray_direction: 机械臂末端光线的方向向量

        Returns:
            compensation_angles: 需要补偿的角度 [x_angle, y_angle, z_angle]（度）
        """
        # 理想方向（电子舱连线方向）
        ideal_direction = np.array([0, 0, -1])

        # 当前方向
        current_direction = ray_direction / np.linalg.norm(ray_direction)

        # 计算旋转矩阵将当前方向旋转到理想方向
        v = np.cross(current_direction, ideal_direction)
        s = np.linalg.norm(v)
        c = np.dot(current_direction, ideal_direction)

        if s < 1e-6:
            # 方向相同或相反，不需要旋转
            return np.array([0.0, 0.0, 0.0])

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)

        # 从旋转矩阵中提取欧拉角（ZYX顺序）
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        # 转换为角度
        compensation_angles = np.degrees([roll, pitch, yaw])
        return compensation_angles

    def check_error_threshold(self, alignment_error, perpendicularity_error):
        """
        检查误差是否超过阈值

        Returns:
            is_within_threshold: 是否在阈值内
            max_error: 最大误差
            status_text: 状态文本
            status_color: 状态颜色
        """
        max_error = max(alignment_error, perpendicularity_error)

        if max_error <= self.error_threshold:
            return True, max_error, "✅ 补偿成功！", "lightgreen"
        else:
            return False, max_error, "❌ 需要继续修正", "lightcoral"

    def init_visualization(self):
        """初始化可视化界面"""
        # 创建图形窗口（三列布局）
        self.fig = plt.figure(figsize=(20, 8))
        self.fig.canvas.manager.set_window_title('机械臂末端位置、光线与误差可视化')

        # 使用GridSpec创建三个子图
        gs = GridSpec(1, 3, width_ratios=[1, 2, 1])

        # 左侧：位置坐标显示
        self.ax_left = self.fig.add_subplot(gs[0])
        self.ax_left.set_axis_off()  # 隐藏坐标轴

        # 中间：3D光线可视化
        self.ax_middle = self.fig.add_subplot(gs[1], projection='3d')

        # 右侧：误差显示
        self.ax_right = self.fig.add_subplot(gs[2])
        self.ax_right.set_axis_off()  # 隐藏坐标轴

        # 设置3D坐标轴标签
        self.ax_middle.set_xlabel('X (mm)')
        self.ax_middle.set_ylabel('Y (mm)')
        self.ax_middle.set_zlabel('Z (mm)')
        self.ax_middle.set_title('机械臂末端光线与舱体中心线')

        # 设置坐标轴范围（根据机械臂尺寸调整）
        self.ax_middle.set_xlim([-1000, 1000])
        self.ax_middle.set_ylim([-1000, 1000])
        self.ax_middle.set_zlim([0, 1000])

        # 左侧：创建文本框
        self.position_text = self.ax_left.text(0.5, 0.8, "末端位置: (0.0000, 0.0000, 0.0000) mm",
                                               fontsize=16, ha='center', va='center',
                                               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        # 目标件孔位置显示
        self.hole_position_text = self.ax_left.text(0.5, 0.7, "目标件孔位置: (0.0000, 0.0000, 0.0000) mm",
                                                    fontsize=16, ha='center', va='center',
                                                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        # 时间戳显示
        self.timestamp_text = self.ax_left.text(0.5, 0.6, "时间: 等待数据...",
                                                fontsize=14, ha='center', va='center',
                                                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        # 中间：绘制舱体中心线（静态）
        cabin_start, cabin_end = self.get_cabin_centerline()
        self.cabin_line, = self.ax_middle.plot([cabin_start[0], cabin_end[0]],
                                               [cabin_start[1], cabin_end[1]],
                                               [cabin_start[2], cabin_end[2]],
                                               'g-', linewidth=3, label='舱体中心线')

        # 标记舱体中心点
        self.ax_middle.scatter([self.cabin_center[0]], [self.cabin_center[1]], [self.cabin_center[2]],
                               c='green', s=50, marker='o', label='舱体中心')

        # 中间：绘制定位孔连线（静态）
        pos_start, pos_end = self.get_positioning_line()
        self.pos_line, = self.ax_middle.plot([pos_start[0], pos_end[0]],
                                             [pos_start[1], pos_end[1]],
                                             [pos_start[2], pos_end[2]],
                                             'b-', linewidth=2, label='定位孔连线')

        # 标记定位孔点
        self.ax_middle.scatter([self.positioning_hole[0]], [self.positioning_hole[1]], [self.positioning_hole[2]],
                               c='blue', s=50, marker='s', label='定位孔')

        # 中间：初始化机械臂末端光线
        self.ray_line, = self.ax_middle.plot([], [], [], 'r-', linewidth=2, label='机械臂末端光线')
        self.ray_start_point, = self.ax_middle.plot([], [], [], 'ro', markersize=5, label='光线起点')
        self.ray_end_point, = self.ax_middle.plot([], [], [], 'rx', markersize=8, label='光线终点')

        # 初始化目标件定位孔连线（动态）
        self.dynamic_hole_line, = self.ax_middle.plot([], [], [], 'c-', linewidth=2, label='目标件定位孔连线')
        self.dynamic_hole_point, = self.ax_middle.plot([], [], [], 'co', markersize=5, label='目标件定位孔')

        # 添加图例
        self.ax_middle.legend()

        # 右侧：创建误差显示文本框
        self.alignment_error_text = self.ax_right.text(0.5, 0.85, "同轴度误差: 0.0000 mm",
                                                       fontsize=12, ha='center', va='center',
                                                       bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                                                 alpha=0.8))

        self.perpendicularity_error_text = self.ax_right.text(0.5, 0.75, "垂直度误差: 0.0000 mm",
                                                              fontsize=12, ha='center', va='center',
                                                              bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                                                        alpha=0.8))

        self.vertical_axis_error_text = self.ax_right.text(0.5, 0.65, "垂直轴偏差: 0.0000°",
                                                           fontsize=12, ha='center', va='center',
                                                           bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                                                     alpha=0.8))

        # 添加误差阈值显示
        self.threshold_text = self.ax_right.text(0.5, 0.55, f"误差阈值: {self.error_threshold} mm",
                                                 fontsize=10, ha='center', va='center',
                                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

        # 添加误差状态显示
        self.error_status_text = self.ax_right.text(0.5, 0.45, "等待数据...",
                                                    fontsize=12, ha='center', va='center',
                                                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        self.displacement_text = self.ax_right.text(0.5, 0.35, "位移修正: (0.00, 0.00, 0.00) mm",
                                                    fontsize=10, ha='center', va='center',
                                                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue",
                                                              alpha=0.8))

        # 同轴度补偿角度显示
        self.alignment_comp_text = self.ax_right.text(0.5, 0.25, "同轴度补偿角度: (0.00°, 0.00°, 0.00°)",
                                                      fontsize=10, ha='center', va='center',
                                                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                                                                alpha=0.8))

        # 垂直度补偿角度显示
        self.perpendicularity_comp_text = self.ax_right.text(0.5, 0.15, "垂直度补偿角度: (0.00°, 0.00°, 0.00°)",
                                                             fontsize=10, ha='center', va='center',
                                                             bbox=dict(boxstyle="round,pad=0.5",
                                                                       facecolor="lightyellow", alpha=0.8))

        plt.tight_layout()
        plt.ion()  # 开启交互模式
        plt.show()

    def update_visualization(self, record):
        """更新可视化界面"""
        joint_angles_rad = record['joint_angles_rad']
        position, rotation_matrix, _ = self.get_end_effector_pose(joint_angles_rad)
        z = position[2]  # 获取Z坐标

        # 获取末端光线
        ray_start, ray_end, direction = self.get_end_effector_ray(joint_angles_rad)

        # 计算目标件定位孔的世界坐标
        hole_world = position + rotation_matrix @ self.positioning_hole_local

        # 计算目标件连线向量（从定位孔到末端中心）
        target_line_vector = position - hole_world

        # 计算垂直轴偏差
        vertical_axis_error = self.calculate_vertical_axis_error(target_line_vector)

        # 初始化误差和修正量为None
        alignment_error = None
        alignment_angle = None
        perpendicularity_error = None
        perpendicularity_angle = None
        displacement_vector = None
        alignment_comp_angles = None
        perpendicularity_comp_angles = None
        is_within_threshold = False
        max_error = 0
        status_text = ""
        status_color = "white"

        # 检查Z坐标是否在0-300范围内
        if 0 <= z <= 300:
            # 计算同轴度误差
            alignment_error, alignment_angle = self.calculate_alignment_error(direction)

            # 计算垂直度误差
            perpendicularity_error, perpendicularity_angle = self.calculate_perpendicularity_error(direction)

            # 计算位移修正向量
            displacement_vector = self.calculate_displacement_vector(direction)

            # 计算同轴度补偿角度
            alignment_comp_angles = self.calculate_alignment_compensation_angles(direction)

            # 计算垂直度补偿角度
            perpendicularity_comp_angles = self.calculate_perpendicularity_compensation_angles(direction)

            # 检查误差阈值
            is_within_threshold, max_error, status_text, status_color = self.check_error_threshold(
                alignment_error, perpendicularity_error)

            # 存储计算值
            self.last_alignment_comp_angles = alignment_comp_angles
            self.last_perpendicularity_comp_angles = perpendicularity_comp_angles
            self.last_vertical_axis_error = vertical_axis_error
        else:
            # 存储默认值
            self.last_alignment_comp_angles = None
            self.last_perpendicularity_comp_angles = None
            self.last_vertical_axis_error = None

        # 存储位置信息
        self.last_position = position
        self.last_hole_world = hole_world

        # 更新左侧文本框内容
        self.position_text.set_text(f"末端位置: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) mm")
        self.hole_position_text.set_text(
            f"目标件孔位置: ({hole_world[0]:.2f}, {hole_world[1]:.2f}, {hole_world[2]:.2f}) mm")
        self.timestamp_text.set_text(f"时间: {record['timestamp']}")

        # 更新右侧显示
        # 检查Z坐标是否在0-300范围内
        if 0 <= z <= 300:
            # 更新右侧误差显示
            self.alignment_error_text.set_text(f"同轴度误差: {alignment_error:.4f} mm")
            self.perpendicularity_error_text.set_text(f"垂直度误差: {perpendicularity_error:.4f} mm")
            self.vertical_axis_error_text.set_text(f"垂直轴偏差: {vertical_axis_error:.4f}°")
            self.error_status_text.set_text(status_text)
            self.error_status_text.set_bbox(dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.8))

            if is_within_threshold:
                self.displacement_text.set_text("误差在阈值范围内，无需位移修正")
            else:
                self.displacement_text.set_text(
                    f"位移修正: ({displacement_vector[0]:.2f}, {displacement_vector[1]:.2f}, {displacement_vector[2]:.2f}) mm")

            # 更新补偿角度显示
            self.alignment_comp_text.set_text(
                f"同轴度补偿角度: ({alignment_comp_angles[0]:.2f}°, {alignment_comp_angles[1]:.2f}°, {alignment_comp_angles[2]:.2f}°)")
            self.perpendicularity_comp_text.set_text(
                f"垂直度补偿角度: ({perpendicularity_comp_angles[0]:.2f}°, {perpendicularity_comp_angles[1]:.2f}°, {perpendicularity_comp_angles[2]:.2f}°)")
        else:
            # Z坐标不在0-300范围内，显示提示信息
            self.alignment_error_text.set_text("同轴度误差: Z坐标超出范围")
            self.perpendicularity_error_text.set_text("垂直度误差: Z坐标超出范围")
            self.vertical_axis_error_text.set_text("垂直轴偏差: Z坐标超出范围")
            self.error_status_text.set_text("Z坐标不在[0,300]mm内")
            self.error_status_text.set_bbox(dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
            self.displacement_text.set_text("位移修正: 不计算")
            self.alignment_comp_text.set_text("同轴度补偿角度: Z坐标超出范围")
            self.perpendicularity_comp_text.set_text("垂直度补偿角度: Z坐标超出范围")

        # 更新中间光线位置
        self.ray_line.set_data([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]])
        self.ray_line.set_3d_properties([ray_start[2], ray_end[2]])

        # 更新光线端点
        self.ray_start_point.set_data([ray_start[0]], [ray_start[1]])
        self.ray_start_point.set_3d_properties([ray_start[2]])

        self.ray_end_point.set_data([ray_end[0]], [ray_end[1]])
        self.ray_end_point.set_3d_properties([ray_end[2]])

        # 更新目标件定位孔连线
        self.dynamic_hole_line.set_data([hole_world[0], position[0]], [hole_world[1], position[1]])
        self.dynamic_hole_line.set_3d_properties([hole_world[2], position[2]])

        # 更新目标件定位孔点
        self.dynamic_hole_point.set_data([hole_world[0]], [hole_world[1]])
        self.dynamic_hole_point.set_3d_properties([hole_world[2]])

        # 重绘图形
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def print_dh_parameters(self):
        """打印DH参数表"""
        print("\n越疆机械臂SDH参数表:")
        print("关节\t a(mm)\t\t alpha(rad)\t d(mm)\t\t theta_offset(rad)")
        print("-" * 70)

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params, 1):
            print(f"{i}\t {a:.2f}\t\t {alpha:.4f}\t\t {d:.2f}\t\t {theta_offset:.4f}")

        print(f"\n基座位置: {self.base_position} mm")
        print(f"基座姿态: {self.base_orientation} (roll, pitch, yaw in radians)")

        # 打印舱体信息
        print("\n舱体中心位置: ", self.cabin_center)
        print("舱体中心线长度: ±", self.cabin_centerline_length, "mm")
        print("定位孔位置: ", self.positioning_hole)
        print("定位线终点: ", self.positioning_line_end)
        print("机械臂末端光线长度: ±", self.end_effector_ray_length, "mm")
        print("目标件定位孔位置: ", self.positioning_hole_local, "mm (相对于目标件中心)")
        print(f"误差阈值: {self.error_threshold} mm")


# ------------------ 主运行逻辑：把原来发送到两个 TCP 服务器的位置改为发送到 UnityTcpClient ------------------

def run_websocket_client(client):
    """运行WebSocket客户端"""
    asyncio.run(client.receive_data())


def main():
    """主函数"""
    print("机械臂实时数据可视化系统")
    print("=" * 50)

    # 1. 创建越疆机械臂模型（单位：毫米）
    print("\n根据越疆参数表创建机械臂模型...")

    # 设置基座位置和姿态
    base_position = [0, 0, 0]  # 世界坐标系原点 [x, y, z] (单位：毫米)
    base_orientation = [0, 0, 0]  # 基座姿态 [roll, pitch, yaw] (弧度)

    # 创建越疆机械臂实例（使用SDH参数）
    robot = YueJiangRobot(
        dh_params=None,
        base_position=base_position,
        base_orientation=base_orientation
    )

    # 打印参数信息
    robot.print_dh_parameters()

    # 2. 创建并启动WebSocket客户端 - 直接使用默认值
    host = 'localhost'  # 直接使用默认值，不需要用户输入
    port = 8080  # 直接使用默认值，不需要用户输入

    ws_client = SocketClient(host=host, port=port)

    # 在单独线程中运行WebSocket客户端
    client_thread = threading.Thread(target=run_websocket_client, args=(ws_client,))
    client_thread.daemon = True
    client_thread.start()
    print(f"WebSocket客户端已启动，连接到 {host}:{port}，等待数据...")

    # 3. 创建并启动 Unity TCP 客户端（可选）
    unity_client = None
    unity_host = '192.168.43.2'  # 直接使用默认值
    unity_port = 3333  # 直接使用默认值
    unity_client = UnityTcpClient(host=unity_host, port=unity_port, reconnect_interval=1.0)
    unity_client.start()
    print(f"Unity TCP客户端已启动，连接到 {unity_host}:{unity_port}")

    # 4. 初始化可视化界面
    robot.init_visualization()
    print("可视化界面已初始化，等待数据...")

    # 5. 主循环：处理接收到的数据
    try:
        while True:
            # 获取缓存中最新的数据
            record = ws_client.get_latest_data()

            if record:
                # 更新可视化
                robot.update_visualization(record)

                # 构建统一的数据帧（合并原 TCPServer 与 ErrorCompensationTCPServer 的数据格式）
                position = robot.last_position
                hole_world = robot.last_hole_world

                if position is None or hole_world is None:
                    # 若尚未计算出位置，跳过发送
                    continue

                z = position[2]

                if 0 <= z <= 300:  # 有效范围：发送补偿信息
                    data_frame = {
                        'timestamp': record['timestamp'],
                        'end_effector_position': position.tolist(),
                        'target_hole_position': hole_world.tolist(),
                        'vertical_axis_error': robot.last_vertical_axis_error if hasattr(robot,
                                                                                         'last_vertical_axis_error') and robot.last_vertical_axis_error is not None else 0.0,
                        'alignment_comp_angles': robot.last_alignment_comp_angles.tolist() if robot.last_alignment_comp_angles is not None else [
                            0, 0, 0],
                        'perpendicularity_comp_angles': robot.last_perpendicularity_comp_angles.tolist() if robot.last_perpendicularity_comp_angles is not None else [
                            0, 0, 0]
                    }
                else:
                    # 非有效范围则只发送基础位置信息
                    data_frame = {
                        'timestamp': record['timestamp'],
                        'end_effector_position': position.tolist(),
                        'target_hole_position': hole_world.tolist()
                    }

                # 发送到 Unity（如果启用）
                if unity_client:
                    unity_client.send(data_frame)

            plt.pause(0.0125)  # 短暂暂停以允许界面更新

    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        print("正在关闭客户端与资源...")
        ws_client.stop()
        if unity_client:
            unity_client.stop()
        print("程序已停止")


if __name__ == "__main__":
    main()