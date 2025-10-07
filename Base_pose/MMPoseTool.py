# 导入必要的库
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torchvision import transforms
from mmcv import Config
from mmdet.apis import inference_detector, init_detector, async_inference_detector
import matplotlib.pyplot as plt
import urllib
import base64
import json
from PIL import Image, ImageDraw
import asyncio

def get_config_path(filename="FasterRCNN.py"):
    # 获取外部运行py的绝对路径
    cwd = os.path.dirname(os.getcwd())
    # 获取当前文件的绝对路径
    file_dirname = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(
        file_dirname, 'configs', filename)
    return config_path
def get_ckpt_path(filename="FasterRCNN-pose.pth"):
    # 获取外部运行py的绝对路径
    cwd = os.path.dirname(os.getcwd())
    # 获取当前文件的绝对路径
    file_dirname = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(
        file_dirname, 'checkpoints/pretrain_model', filename)
    return ckpt_path
class MMPoseTool:
    def __init__(self, mmpose_det_config_file="FasterRCNN.py",
                 mmpose_det_ckpt_file="FasterRCNN-pose.pth",
                 mmpose_pose_config_file="SCNet.py",
                 mmpose_pose_ckpt_file="SCNet.pth"):
        self.mmpose_det_config_path = get_config_path(mmpose_det_config_file)
        self.mmpose_det_ckpt_path = get_ckpt_path(mmpose_det_ckpt_file)
        self.mmpose_pose_config_path =get_config_path(mmpose_pose_config_file)
        self.mmpose_pose_ckpt_path = get_ckpt_path(mmpose_pose_ckpt_file)
        from mmpose.models import build_posenet
        from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                                 vis_pose_result, process_mmdet_results, vis_pose_tracking_result)

        self.process_mmdet_results = process_mmdet_results
        self.inference_top_down_pose_model = inference_top_down_pose_model
        self.vis_pose_result = vis_pose_result
        self.vis_pose_tracking_result = vis_pose_tracking_result

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.det_model = init_detector(self.mmpose_det_config_path,
                                       self.mmpose_det_ckpt_path,
                                       device=self.device)

        self.pose_model = init_pose_model(self.mmpose_pose_config_path,
                                          self.mmpose_pose_ckpt_path,
                                          device=self.device)
        self.expected_keypoint_count = 17

    def get_features(self, image_path):
        if isinstance(image_path[0], np.ndarray):
            img = image_path
        else:
            img = cv2.imread(image_path)
        w = img.shape[0]
        h = img.shape[1]
        feature = self._extract_pose_features_mmpose(img)
        return self.normalized(feature, w, h)

    def see_features(self, image_path, pose_features):
        return self._visualize_keypoints_mmpose(image_path, pose_features)

    def process_images_in_folders(self, root_folder):
        return self._process_images_in_folders(root_folder)

    def _process_images_in_folders(self, root_folder):
        feature_data = []
        labels = []

        root_content = [f.path for f in os.scandir(root_folder)]
        is_test_data = any([os.path.isfile(path) for path in root_content])

        if is_test_data:
            # 处理测试数据
            image_files = [f.path for f in os.scandir(root_folder) if f.is_file() and (f.name.endswith(".jpg") or f.name.endswith(".png") or f.name.endswith(".jpeg"))]
            total_images = len(image_files)

            print("Processing images...")
            with tqdm(total=total_images) as progress_bar:
                for filename in image_files:
                    image_path = filename
                    pose_features = self.extract_pose_features(image_path)
                    if pose_features:
                        feature_data.extend(pose_features)
                    progress_bar.update(1)
            print("测试数据特征提取完毕")
        else:
            # 处理训练数据
            subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
            total_images = sum([len([name for name in os.listdir(folder) if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg")]) for folder in subfolders])

            print("Processing images...")
            with tqdm(total=total_images) as progress_bar:
                for i, folder in enumerate(subfolders):
                    for filename in os.listdir(folder):
                        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                            image_path = os.path.join(folder, filename)
                            pose_features = self.extract_pose_features(image_path)
                            if pose_features:
                                feature_data.extend(pose_features)
                                labels.extend([i] * len(pose_features))
                            progress_bar.update(1)
            print("训练数据特征提取完毕")

        return np.array(feature_data), np.array(labels)

    #  输入图片路径，输出预测姿态类别
    def _extract_pose_features_mmpose(self, image_path):
        # inference detection
        mmdet_results = inference_detector(self.det_model, image_path)

        # extract person (COCO_ID=1) bounding boxes from the detection results
        person_results = self.process_mmdet_results(mmdet_results, cat_id=1)

        # inference top-down pose model
        self.pose_results, returned_outputs = self.inference_top_down_pose_model(
            self.pose_model, image_path, person_results, format='xyxy', dataset=self.pose_model.cfg.data.test.type
        )

        # 提取姿势特征并返回
        pose_features = []
        for res in self.pose_results:
            keypoints = res['keypoints']
            if len(keypoints) == self.expected_keypoint_count:
                keypoints_flat = []
                for point in keypoints:
                    x, y, _ = point
                    keypoints_flat.extend([x, y])
                pose_features.append(keypoints_flat)
        return pose_features

    def _visualize_keypoints_mmpose(self, image_path, pose_features):
        vis_result = self.vis_pose_result(self.pose_model,
                                      image_path,
                                      self.pose_results,
                                      dataset=self.pose_model.cfg.data.test.type,
                                      show=False,
                                      radius=20,
                                      thickness=10)

        return cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB)

    def crop_and_save_detected_objects(self, root_folder, output_folder):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
        total_images = sum([len([name for name in os.listdir(folder) if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg")]) for folder in subfolders])

        print("Processing images...")
        with tqdm(total=total_images) as progress_bar:
            for i, folder in enumerate(subfolders):
                folder_name = os.path.basename(folder)
                output_subfolder = os.path.join(output_folder, folder_name)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                for filename in os.listdir(folder):
                    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                        image_path = os.path.join(folder, filename)
                        mmdet_results = inference_detector(self.det_model, image_path)
                        person_results = self.process_mmdet_results(mmdet_results, cat_id=1)
                        img = Image.open(image_path)
                        width, height = img.size
                        padding_ratio = 0.1 # this means the padding is 10% of the width/height of the bounding box
                        for j, person in enumerate(person_results):
                            bbox = person['bbox'][:4]
                            bbox[0] = max(0, bbox[0] - padding_ratio * (bbox[2] - bbox[0])) # left
                            bbox[1] = max(0, bbox[1] - padding_ratio * (bbox[3] - bbox[1])) # upper
                            bbox[2] = min(width, bbox[2] + padding_ratio * (bbox[2] - bbox[0])) # right
                            bbox[3] = min(height, bbox[3] + padding_ratio * (bbox[3] - bbox[1])) # lower
                            crop_img = img.crop(bbox)
                            crop_img.save(os.path.join(output_subfolder, f"{filename}_crop_{j}.jpg"))
                        progress_bar.update(1)
        print("Crop and save completed")

    def normalized(self,feature,width,height):
        norm_feature = []
        for j in feature:
            temp = []
            for i in range(len(j)):
                if i%2==0:
                    temp.append(j[i]/width)
                else:
                    temp.append(j[i]/height)
            norm_feature.append(temp)
        return norm_feature