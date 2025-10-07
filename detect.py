from Base_pose import MediapipePose
import matplotlib.pyplot as plt
from BaseML import Classification as cls
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np

# 初始化识别信息和分数
y_pred21 = None
scores = None
# 指定一张新图片
image_path = 'picture/test1.jpg'
result_path = 'picture/result_RandomForest.jpg'
# 实例化MMPoseTool类，用于后续提取图像的人体姿态关键点
pose = MediapipePose()
# 提取图像中的姿态关键点特征
pose_features = pose.get_features(image_path)
model21 = cls('RandomForest')

# 载入已保存模型
model21.load('checkpoints/mediapipe_pose_RandomForest_mode.pkl')
if len(pose_features) == 0:  # 如果特征数据为空则不进行模型预测
    print("未提取到关键点")
else:
    y_pred21 = model21.inference(pose_features)[0]
    scores = model21.inference(pose_features)[1]

# 指定分类标签
label = ['handsup', 'hello', 'hug', 'jump','kneel','run','sit','stand','walk']
print('预测结果为：', label[int(y_pred21[0])])

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


# 将提取的姿态关键点绘制到图像上，以便可视化
visualized_image = pose.see_features(image_path, pose_features)
# 获取图像宽高
h, w = visualized_image.shape[0], visualized_image.shape[1]
# 将识别的类别显示在图像上
visualized_image = paint_chinese_opencv(visualized_image, label[int(y_pred21[0])], (h/2, 5), (255, 0, 255), 150)

# 显示每个分数
for i, score in enumerate(scores[0]):
    # 将分数转换为字符串
    score_text = f"{score * 100:.2f}%"  # 乘以100并添加百分号
    # 设定文本的位置，y 轴每次增加 50 像素
    position = (50, i * 50)  # (x, y) 坐标
    visualized_image = paint_chinese_opencv(visualized_image, label[i] + "：" + score_text, position, (255, 0, 255), 50)

# 显示处理后的图片
plt.imshow(visualized_image)
plt.show()

#保存处理后的图片
if cv2.imwrite(result_path, visualized_image):
    print(f'保存成功，文件已保存在{result_path}下')
else:
    print(f'保存失败，请检查{result_path}保存路径、权限与格式是否正确')
