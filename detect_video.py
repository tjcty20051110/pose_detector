# mediapipe人工智能工具包
import mediapipe as mp
import numpy as np
import threading
from BaseML import Classification as cls

# 导入opencv-python
import cv2
import time
from PIL import Image, ImageFont, ImageDraw
import json

'''
# 服务器的主机和端口
import socket
server_host = '127.0.0.1'
server_port = 6666
# 创建一个TCP套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接到服务器
client_socket.connect((server_host, server_port))
'''
send_flag = False
first_time = time.perf_counter()
# 指定分类标签
label = ['handsup', 'hello', 'hug', 'jump','kneel','run','squat','stand','walk']
# 载入已保存模型
model21 = cls('RandomForest')
model21.load('checkpoints/mediapipe_pose_RandomForest_mode.pkl')
str_pos = "stand"
y_pred21 = None
# 导入solution
mp_pose = mp.solutions.pose

# # 导入绘图函数
mp_drawing = mp.solutions.drawing_utils

# 导入模型
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=2,            # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,         # 是否平滑关键点
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)   # 追踪阈值


def paint_chinese_opencv(im, chinese, pos, color, size):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('NotoSansCJK-Bold.ttc', int(size), encoding="utf-8")
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    # if not isinstance(chinese,unicode):
    # chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, fillColor, font)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle


def get_pos(pose_features):
    global y_pred21
    pose_features = [pose_features]
    y_pred21 = model21.inference(pose_features)[0]
    scores = model21.inference(pose_features)[1]
    if scores is not None:
        return label[int(y_pred21[0])], scores[0]
    else:
        return label[int(y_pred21[0])], None



def process_frame(img):
    global first_time
    global str_pos
    global send_flag
    global y_pred21
    # 记录该帧开始处理的时间
    start_time = time.time()

    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    '''
    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)
    '''

    # 存储关键点的具体位置
    data = []

    results = pose.process(img)

    if results and results.pose_landmarks:  # 若检测出人体关键点

        # 可视化关键点及骨架连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i in range(33):  # 遍历所有33个关键点，可视化

            # 获取该关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z
            # 获得data的位置
            data.extend([results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, results.pose_landmarks.landmark[i].z])
            radius = 10
            if i == 0:  # 鼻尖
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [11,12]:  # 肩膀
                img = cv2.circle(img,(cx,cy), radius, (223,155,6), -1)
            elif i in [23,24]:  # 髋关节
                img = cv2.circle(img,(cx,cy), radius, (1,240,255), -1)
            elif i in [13,14]:  # 胳膊肘
                img = cv2.circle(img,(cx,cy), radius, (140,47,240), -1)
            elif i in [25,26]:  # 膝盖
                img = cv2.circle(img,(cx,cy), radius, (0,0,255), -1)
            elif i in [15,16,27,28]:  # 手腕和脚腕
                img = cv2.circle(img,(cx,cy), radius, (223,155,60), -1)
            elif i in [17,19,21]:  # 左手
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            elif i in [18,20,22]:  # 右手
                img = cv2.circle(img,(cx,cy), radius, (16,144,247), -1)
            elif i in [27,29,31]:  # 左脚
                img = cv2.circle(img,(cx,cy), radius, (29,123,243), -1)
            elif i in [28,30,32]:  # 右脚
                img = cv2.circle(img,(cx,cy), radius, (193,182,255), -1)
            elif i in [9,10]:  # 嘴
                img = cv2.circle(img,(cx,cy), radius, (205,235,255), -1)
            elif i in [1,2,3,4,5,6,7,8]:  # 眼及脸颊
                img = cv2.circle(img,(cx,cy), radius, (94,218,121), -1)
            else:  # 其它关键点
                img = cv2.circle(img,(cx,cy), radius, (0,255,0), -1)
        str_pos_tmp, scores = get_pos(data)
        current_time = time.perf_counter()
        if current_time - first_time >= 1.2:  # 检查是否已经过了1.2秒
            str_pos, scores = get_pos(data)
            if str_pos != str_pos_tmp:
                str_pos = "stand"
            first_time = current_time  # 重置起始时间
            if str_pos != "stand":
                send_flag = True
        if scores is not None:
            score_text = f"{scores[int(y_pred21[0])] * 100:.2f}%"  # 乘以100并添加百分号
        else:
            score_text = ''
        img = paint_chinese_opencv(img, str_pos + "：" + score_text, (h/2, 5), (255, 0, 0), h/10)
    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)

    scaler = 1
    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img

if __name__ == '__main__':
    video_path = 'video/run.mp4'
    result_path = 'video/result_run.mp4'
    cap = cv2.VideoCapture(video_path)

    # 获取视频参数（用于保存输出）
    fps = cap.get(cv2.CAP_PROP_FPS)              # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# 高度

    # 初始化视频写入器
    # 编码器选择'mp4v'（生成MP4文件，兼容性好）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    # 检查写入器是否初始化成功
    if not out.isOpened():
        print(f"错误：无法创建输出视频 {result_path}")
        cap.release()
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret==False:
            break

        #处理每一帧
        result_frame = process_frame(frame)

        #显示结果
        cv2.imshow('result_frame', result_frame)
        # 处理当前帧
        processed_frame = process_frame(frame)
        # 写入处理后的帧到输出视频
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
