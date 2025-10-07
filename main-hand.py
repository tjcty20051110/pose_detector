import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)  # 定义一个摄像头
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)  # 改变识别点的颜色和粗细
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)  # 改变识别线的颜色和粗细
pTime = 0
cTime = 0


def get_str_guester(up_fingers, list_lms):
    if len(up_fingers) == 1 and up_fingers[0] == 8:
        v1 = list_lms[6] - list_lms[7]
        v2 = list_lms[8] - list_lms[7]
        angle = get_angle(v1, v2)
        if angle < 160:
            str_word = "9"
        else:
            str_word = "1"
    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        str_word = "Good"
    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        str_word = "Bad"
    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        str_word = "FXXX"
    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_word = "2"
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        str_word = "6"
    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        str_word = "8"
    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        str_word = "3"
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:
        dis_8_12 = list_lms[8, :] - list_lms[12, :]
        dis_8_12 = np.sqrt(np.dot(dis_8_12, dis_8_12))

        dis_4_12 = list_lms[4, :] - list_lms[12, :]
        dis_4_12 = np.sqrt(np.dot(dis_4_12, dis_4_12))

        if dis_4_12/(dis_8_12 + 1) < 3:
            str_word = "7"
        elif dis_4_12/(dis_8_12 + 1) > 5:
            str_word = "Gun"
        else:
            str_word = "7"
    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        str_word = "ROCK"
    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[3] == 20:
        str_word = "4"
    elif len(up_fingers) == 5:
        str_word = "5"
    elif len(up_fingers) == 0:
        str_word = "10"
    else:
        str_word = " "
    return str_word


def get_angle(v1, v2):
    angle = np.dot(v1, v2)/(np.sqrt(np.sum(v1*v1))*(np.sqrt(np.sum(v2*v2))))
    angle = np.arccos(angle)/3.14*180

    return angle


while True:
    ret, img = cap.read()  # 读取摄像头图像
    list_lms = []  # 创建一个存储所有识别点的list数组
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像由BGR转为RGB
        result = hands.process(imgRGB)  # 得到手部检测结果
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]  # 获取图像的高度
        imgWidth = img.shape[1]  # 获取图像的宽度
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # 在图像上标注识别点，参数（图像，标注文字，文字坐标，字体，大小，颜色，粗细）
                    if i == 4 or i == 8 or i == 12:
                        cv2.circle(img, (xPos, yPos), 10, (255, 0, 0), cv2.FILLED)
                    list_lms.append([int(xPos), int(yPos)])
                    # print(i, xPos, yPos)
            list_lms = np.array(list_lms, dtype=np.int32)
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
            hull = cv2.convexHull(list_lms[hull_index, :])
            cv2.polylines(img, [hull], True, (0, 255, 255), 2)

            n_fig = -1
            ll = [4, 8, 12, 16, 20]
            up_fingers = []
            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    up_fingers.append(i)
            str_guester = get_str_guester(up_fingers, list_lms)
            cv2.putText(img, ' %s' %(str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break