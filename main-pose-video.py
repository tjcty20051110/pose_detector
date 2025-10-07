# main-pose-video.py：视频文件处理模块
import cv2
import os
from common import process_frame


def process_video(file_path, save_path=None):
    """处理视频文件并可选保存结果"""
    # 打开视频
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {file_path}")
        return

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入器（如需保存）
    out = None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"警告：无法创建视频文件 {save_path}，将不保存")
            out = None

    # 逐帧处理
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # 视频结束

        frame = cv2.flip(frame, 1)  # 镜像处理
        processed_frame = process_frame(frame)

        # 显示帧
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        cv2.imshow('video', processed_frame)

        # 写入保存（如果启用）
        if out:
            out.write(processed_frame)

        # 按q退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    # 释放资源
    cap.release()
    if out:
        out.release()
        print(f"视频已保存至：{save_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 示例：处理视频并保存
    process_video(
        file_path="video/run.mp4",  # 输入视频路径
        save_path="video/output_run.mp4"  # 保存路径（可选）
    )