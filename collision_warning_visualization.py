import json
import time
import threading
import queue
import socket
import numpy as np


# ------------------- Numpy 支持的 JSON Encoder -------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ------------------- Unity TCP 客户端 -------------------
class UnityTcpClient:
    """
    Python -> Unity 的 TCP 客户端。
    - 自动重连
    - 发送队列
    """

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

                sock.settimeout(None)
                with self._lock:
                    self._sock = sock
                print(f"[{self.name}] ✅ 已连接到 Unity {self.host}:{self.port}")

                while self._running:
                    try:
                        data = self.send_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    try:
                        json_data = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)
                        message = (json_data + '\n').encode('utf-8')
                        with self._lock:
                            if self._sock:
                                self._sock.sendall(message)
                    except Exception as e:
                        print(f"[{self.name}] 发送失败: {e}，将尝试重连并重新入队该数据")
                        try:
                            self.send_queue.put_nowait(data)
                        except:
                            pass
                        with self._lock:
                            try:
                                self._sock.close()
                            except:
                                pass
                            self._sock = None
                        break

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

        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except:
                    pass
                self._sock = None
        print(f"[{self.name}] 客户端线程已停止")

    def send(self, data):
        try:
            self.send_queue.put_nowait(data)
            return True
        except Exception as e:
            print(f"[{self.name}] 入队失败: {e}")
            return False

    def stop(self):
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


# ------------------- AR数据流示例 -------------------
data_stream_example = [
    {
        "timestamp": "2024-06-01T12:00:00Z",
        "type": "human_robot",
        "risk_level": "low",
        "warning_message": "人机碰撞低风险",
        "color": (0, 255, 0)
    },
    {
        "timestamp": "2024-06-01T12:00:00Z",
        "type": "equipment_cabin",
        "risk_level": "low",
        "warning_message": "舱壁碰撞低风险",
        "color": (0, 255, 0)
    },
    {
        "timestamp": "2024-06-01T12:00:00Z",
        "type": "cable_equipment",
        "risk_level": "low",
        "warning_message": "线缆碰撞低风险",
        "color": (0, 255, 0)
    },
]


# ------------------- 模拟发送数据 -------------------
def simulate_data_stream(data, category, tcp_client):
    """模拟数据流输出并发送到 Unity"""
    while True:
        ar_data = data[category]
        # 发送到 Unity
        tcp_client.send(ar_data)
        # 也打印到控制台
        print(f"[模拟发送] {ar_data}")
        time.sleep(0.5)


# ------------------- 主程序 -------------------
if __name__ == "__main__":
    unity_client = UnityTcpClient(host='192.168.43.2', port=3333)
    unity_client.start()

    # 构建三个线程，模拟三种碰撞检测的数据输出
    threads = [
        threading.Thread(target=simulate_data_stream, args=(data_stream_example, 0, unity_client)),
        threading.Thread(target=simulate_data_stream, args=(data_stream_example, 1, unity_client)),
        threading.Thread(target=simulate_data_stream, args=(data_stream_example, 2, unity_client))
    ]
    for t in threads:
        t.start()

    for t in threads:
        t.join()
