import threading
from typing import Optional

class GestureRecognitionInterface:
    """
    手势识别结果接口（线程安全）
    提供对手势识别结果的访问
    """
    def __init__(self):
        self._current_gesture = -1  # 默认-1
        self._prob = 0.0
        self._lock = threading.Lock()
        self._label_names = {
            -1: "无手势",      # 默认手势
            7: "抓握",        # 机械臂抓取零件
            8: "释放",        # 机械臂释放零件
            # 可根据需要扩展其他标签
        }

    def update(self, gesture: int, prob: float):
        """更新当前手势状态（由主循环调用）"""
        with self._lock:
            self._current_gesture = gesture
            self._prob = prob

    def get_current_gesture(self) -> int:
        """
        获取当前手势标签
        返回:
            int: -1(无手势), 7(抓握), 8(释放)
        """
        with self._lock:
            return self._current_gesture

    def get_gesture_name(self) -> str:
        """获取当前手势名称"""
        with self._lock:
            return self._label_names.get(self._current_gesture, f"未知手势({self._current_gesture})")

# 全局接口实例（单例模式）
global_gesture_interface = GestureRecognitionInterface()