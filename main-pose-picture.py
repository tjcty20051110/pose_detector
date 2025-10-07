# main-pose-image.py：单张图片处理模块
import cv2
import os
from common import process_frame,str_pos


def process_image(file_path, save_path=None):
    """处理单张图片并可选保存结果"""
    # 读取图片
    img = cv2.imread(file_path)
    if img is None:
        print(f"错误：无法打开图片 {file_path}")
        return

    # 处理图片（标记为图片模式）
    result_img = process_frame(img, is_image=True)
    print(f"图片识别结果：{str_pos}")

    # 保存结果（如果指定路径）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if cv2.imwrite(save_path, result_img):
            print(f"结果已保存至：{save_path}")
        else:
            print(f"警告：无法保存至 {save_path}")

    # 显示结果
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', result_img)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 示例：处理图片并保存
    process_image(
        file_path="picture/test1.jpg",  # 输入图片路径
        save_path="picture/output.jpg"  # 保存路径（可选）
    )