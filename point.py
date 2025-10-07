from Base_pose import MediapipePose
import os
from tqdm import tqdm
import csv
from BaseDT.dataset import split_tab_dataset
import sys

# 指定类别信息（按照实际需求填写）
class2id = {'handsup':0, 'hello':1, 'hug':2, 'jump':3,'kneel':4,'run':5,'sit':6,'stand':7,'walk':8}


# 定义一个读取训练数据的函数，函数会返回一个包含path文件中所有文件夹路径的数组
def read_folder(path):
    subfolders = []
    for folder in os.scandir(path):           # 该函数用于遍历path路径中的所有文件
        if folder.is_dir():                   # 该方法用于确定文件是不是一个目录
            subfolders.append(folder.path)    # 该方法用于获取文件的路径
    return subfolders


# 定义一个统计图片数量与收集图片路径的函数，函数会返回数组中各个文件夹中的图片总数与一个由图片路径组成的数组
def collect_image(folder_list):
    total = 0
    image_path = []
    for folder in folder_list:
        for name in os.listdir(folder):
            # 通过判断后缀，判断文件是否为图片
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg"):
                total += 1
                image_path.append(os.path.join(folder, name))
    return total, image_path


# 指定训练集图片对应路径
root_folder = 'pose GUI'
# 处理训练数据
subfolders = read_folder(root_folder)
# 计算图片数量
total_images, image_path = collect_image(subfolders)
# 载入数据集
pose = MediapipePose()
feature_data = []  # 存储特征值
labels = []  # 存储标签
filenames = []  # 存储文件名

print("图片姿态关键点提取中")

error_messages = []  # 创建一个列表来收集错误信息
with tqdm(total=total_images, file=sys.stdout) as progress_bar:
    for image in image_path:
        # 生成特征文件
        features = pose.get_features(image)

        # 判断特征提取是否完整
        # 这里的17*2是MMPose的特征点的数量（17个特征的，2个坐标），根据不同模型需要进行修改
        # mediapipe是33个特征，3个坐标
        if features and len(features[0]) == 33 * 3:
            filenames.extend([image] * len(features))  # 因为可能会出现一张图片中有多个姿态，因此在此乘上识别出的姿态数量
            feature_data.extend(features)
            labels.extend([class2id[image.split(os.sep)[-2]]] * len(features))  # 训练集有对应标签，因此可以加上对应标签
        elif len(features) == 0:
            error_messages.append("路径" + image + "的图片未提取到人体姿态关键点")
        else:
            error_messages.append("路径" + image + "的图片未提取到人体姿态关键点个数不全")
        progress_bar.update(1)

print("数据集数据关键点提取完毕")

# 打印出所有的错误信息
for message in error_messages:
    print(message)

# 将提取的关键点特征和标签保存为一个csv
header = ["Path_Filename"]+[f"Feature {i+1}" for i in range(len(feature_data[0]))] + ["Label"]
csv_data = []
for i in range(len(feature_data)):
    csv_data.append(feature_data[i].copy())
    csv_data[-1].append(labels[i])
    csv_data[-1].insert(0, filenames[i])
with open('feature_data/mediapipe_pose.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_data)

# 指定待拆分的csv数据集
path = "feature_data/mediapipe_pose.csv"
# 指定特征数据列、标签列、训练集比重，'normalize=True'表示进行归一化
tx, ty, val_x, val_y = split_tab_dataset(path,data_column=range(1, 100), label_column=100, train_val_ratio=0.8)
