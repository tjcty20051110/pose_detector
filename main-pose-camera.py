# main-pose-camera.py：摄像头实时处理模块
import cv2
import json
from common import process_frame, send_flag, str_pos


def process_camera():
    """处理摄像头实时画面"""
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    # 实时处理循环
    while cap.isOpened():
        # 发送识别结果（非normal状态）
        global send_flag
        if send_flag:
            send_flag = False
            json_data = {"ActionCode": str_pos}
            print(f"识别结果：{json.dumps(json_data)}")
            # 实际发送时取消注释
            # client_socket.send(json.dumps(json_data).encode('utf-8'))

        # 读取帧
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # 镜像操作
        processed_frame = process_frame(frame)

        # 显示画面
        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        cv2.imshow('camera', processed_frame)

        # 按q退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 启动摄像头识别
    process_camera()