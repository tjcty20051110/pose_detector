# common.py：公用函数模块
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2

# 初始化MediaPipe姿态模型
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 全局状态变量（跨模块共享）
send_flag = False
first_time = 0.0
str_pos = "normal"


def paint_chinese_opencv(im, chinese, pos, color):
    """绘制中文字符"""
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype('NotoSansCJK-Bold.ttc', 50, encoding="utf-8")
    except:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img_PIL)
    draw.text(pos, chinese, color, font)
    return cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)


def get_angle(v1, v2):
    """计算两向量夹角（带方向）"""
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)) + 1e-6)
    angle = np.arccos(np.clip(angle, -1.0, 1.0)) / np.pi * 180
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = -angle
    return angle


def get_pos(keypoints):
    """动作识别核心逻辑（7种动作）"""
    keypoints = np.array(keypoints)

    # 提取核心关键点
    left_shoulder, right_shoulder = keypoints[11], keypoints[12]
    left_elbow, right_elbow = keypoints[13], keypoints[14]
    left_wrist, right_wrist = keypoints[15], keypoints[16]
    left_hip, right_hip = keypoints[23], keypoints[24]
    left_knee, right_knee = keypoints[25], keypoints[26]
    left_ankle, right_ankle = keypoints[27], keypoints[28]
    hip_center = (left_hip + right_hip) / 2

    # 1. 蹲下（squat）
    def calc_knee_angle(hip, knee, ankle):
        return get_angle(hip - knee, ankle - knee)
    left_knee_angle = calc_knee_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calc_knee_angle(right_hip, right_knee, right_ankle)
    if (left_knee_angle < 70 and right_knee_angle < 70) and (hip_center[1] > left_knee[1] + 30):
        return "squat"

    # 2. 跳跃（jump）
    feet_off = (left_ankle[1] < left_knee[1] - 40) and (right_ankle[1] < right_knee[1] - 40)
    body_rising = (hip_center[1] < (left_shoulder[1] + right_shoulder[1]) / 2 - 50)
    if feet_off and body_rising:
        return "jump"

    # 3. 拥抱（hug）
    def calc_elbow_angle(shoulder, elbow, wrist):
        return get_angle(shoulder - elbow, wrist - elbow)
    left_elbow_angle = calc_elbow_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calc_elbow_angle(right_shoulder, right_elbow, right_wrist)
    arms_bent = left_elbow_angle < 80 and right_elbow_angle < 80
    wrists_cross = abs(left_wrist[0] - right_wrist[0]) < 80
    if arms_bent and wrists_cross and (left_wrist[1] < left_hip[1]):
        return "hug"

    # 4. 举手（handsup）
    left_up = (left_wrist[1] < left_shoulder[1] - 50) and (calc_elbow_angle(left_shoulder, left_elbow, left_wrist) > 160)
    right_up = (right_wrist[1] < right_shoulder[1] - 50) and (calc_elbow_angle(right_shoulder, right_elbow, right_wrist) > 160)
    if left_up or right_up:
        return "handsup"

    # 5. 挥手（hello）
    left_arm_raised = (left_wrist[1] < left_shoulder[1] + 30) and (left_wrist[1] > left_shoulder[1] - 30)
    right_arm_raised = (right_wrist[1] < right_shoulder[1] + 30) and (right_wrist[1] > right_shoulder[1] - 30)
    left_wave = left_arm_raised and (abs(left_wrist[0] - left_shoulder[0]) > 100)
    right_wave = right_arm_raised and (abs(right_wrist[0] - right_shoulder[0]) > 100)
    if left_wave or right_wave:
        return "hello"

    # 6. 跑步（run）
    step_stride = abs(left_ankle[0] - right_ankle[0]) > 200
    one_foot_off = (left_ankle[1] < left_knee[1] - 20) or (right_ankle[1] < right_knee[1] - 20)
    legs_bent = (left_knee_angle < 140) and (right_knee_angle < 140)
    if step_stride and one_foot_off and legs_bent:
        return "run"

    # 未匹配则为站立
    return "stand"


def process_frame(img, is_image=False):
    """处理单帧图像的公用逻辑"""
    global first_time, str_pos, send_flag
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    h, w = img.shape[0], img.shape[1]

    # 处理图像
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    data = []

    if results.pose_landmarks:
        # 绘制关键点
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            data.append([cx, cy])

            # 关键点染色
            radius = 10
            color_map = {
                0: (0,0,255), 11: (223,155,6), 12: (223,155,6),
                23: (1,240,255), 24: (1,240,255), 13: (140,47,240), 14: (140,47,240),
                25: (0,0,255), 26: (0,0,255), 15: (223,155,60), 16: (223,155,60),
                27: (29,123,243), 28: (193,182,255)
            }
            color = color_map.get(i, (0,255,0))
            img = cv2.circle(img, (cx, cy), radius, color, -1)

        # 动作识别逻辑
        str_pos_tmp = get_pos(data)
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if is_image:
            str_pos = str_pos_tmp  # 图片直接更新
        else:
            if current_time - first_time >= 1.2:  # 视频/摄像头按间隔更新
                str_pos = str_pos_tmp
                if str_pos != str_pos_tmp:
                    str_pos = "normal"
                first_time = current_time
                if str_pos != "normal":
                    send_flag = True

        img = paint_chinese_opencv(img, str_pos, (300, 5), (255, 0, 0))
    else:
        scaler = 1
        img = cv2.putText(
            img, 'No Person', (25*scaler, 100*scaler),
            cv2.FONT_HERSHEY_SIMPLEX, 1.25*scaler, (255,0,255), 2*scaler
        )

    # 计算FPS
    end_time = cv2.getTickCount() / cv2.getTickFrequency()
    FPS = 1 / (end_time - start_time)
    img = cv2.putText(
        img, f'FPS  {int(FPS)}', (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,0,255), 2
    )
    return img

