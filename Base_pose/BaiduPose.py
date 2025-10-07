# 导入必要的库
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torchvision import transforms
from mmcv import Config
from mmdet.apis import inference_detector, init_detector
import matplotlib.pyplot as plt
import urllib
import base64
import json
from PIL import Image, ImageDraw

class BaiduPose:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret

    def get_token(self):
        host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + self.client_id + '&client_secret=' + self.client_secret
        request = urllib.request.Request(host)
        request.add_header('Content-Type', 'application/json; charset=UTF-8')
        response = urllib.request.urlopen(request)
        token_content = response.read()
        if token_content:
            token_info = json.loads(token_content)
            token_key = token_info['access_token']
        return token_key

    def draw_bodys(self, originfilename, bodys, pointsize=5):
        image_origin = Image.open(originfilename)
        draw = ImageDraw.Draw(image_origin)

        for body in bodys:
            for body_part in body['body_parts'].values():
                draw.ellipse((body_part['x'] - pointsize, body_part['y'] - pointsize, body_part['x'] + pointsize, body_part['y'] + pointsize), fill="blue")
            gesture = body['location']
            draw.rectangle((gesture['left'], gesture['top'], gesture['left'] + gesture['width'], gesture['top'] + gesture['height']), outline="red")

        return image_origin

        #image_origin.save(resultfilename, "JPEG")
        #plt.imshow(image_origin)

    def see_features(self, filename, pointsize=5):
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_analysis"
        f = open(filename, 'rb')
        img = base64.b64encode(f.read())

        params = dict()
        params['image'] = img
        params = urllib.parse.urlencode(params).encode("utf-8")

        access_token = self.get_token()
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')
        response = urllib.request.urlopen(request)
        content = response.read()

        if content:
            content = content.decode('utf-8')
            data = json.loads(content)
            result = data['person_info']

            image_origin = self.draw_bodys(filename, result, pointsize)
            
        return image_origin
    
    def process_data(self, main_folder):

        # 判断是否为测试集数据
        root_content = [f.path for f in os.scandir(main_folder)]
        is_test_data = any([os.path.isfile(path) for path in root_content])
        
        features = []
        labels = []

        if is_test_data:
            folders = [('', main_folder)]
        else:
            folders = [(label, os.path.join(main_folder, label)) for label in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, label))]

        image_paths = []
        for label, folder in folders:
            image_files = [image_file for image_file in os.listdir(folder) if image_file.endswith(('.jpg', '.jpeg', '.png'))]
            image_paths += [(label, os.path.join(folder, image_file)) for image_file in image_files]

        total_images = len(image_paths)
        for label, image_path in tqdm(image_paths, total=total_images, desc="Processing images"):
            keypoints = self.extract_keypoints(image_path)
            if keypoints:
                flattened_keypoints = self.flatten_keypoints(keypoints)
                features.append(flattened_keypoints)
                if not is_test_data:
                    labels.append(int(label[-1])-1)

        features_data = np.array(features)
        labels_data = np.array(labels) if not is_test_data else None
        
        if is_test_data:
            print("测试数据特征提取完毕")
        else:
            print("训练数据特征提取完毕")
        # 假设 F 和 L 的形状均为 (M,)
        F = features_data  # 您的特征数组
        L = labels  # 您的标签数组
        N = 42  # 特征的列数

        new_features = []
        new_labels = []

        for features, label in zip(F, L):
            num_rows = len(features) // N
            for i in range(num_rows):
                sub_feature = features[i * N: (i + 1) * N]
                new_features.append(sub_feature)
                new_labels.append(label)

        new_F = np.array(new_features)
        new_L = np.array(new_labels)            

        return (new_F, new_L) if not is_test_data else new_F

    def flatten_keypoints(self, keypoints):
        flattened_keypoints = []
        for body_keypoints in keypoints:
            for _, value in body_keypoints.items():
                flattened_keypoints.extend(value)
        return flattened_keypoints
        
    def get_features(self, image_path):
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_analysis"
        f = open(image_path, 'rb')
        img = base64.b64encode(f.read())

        params = dict()
        params['image'] = img
        params = urllib.parse.urlencode(params).encode("utf-8")

        access_token = self.get_token()
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')
        response = urllib.request.urlopen(request)
        content = response.read()

        if content:
            content = content.decode('utf-8')
            data = json.loads(content)
            #print(data)
            if 'error_code' in data:
                print(image_path,data)
                return [[0]]
            else:
                result = data['person_info']
                keypoints = []
                for body in result:
                    body_keypoints = {}
                    for key, value in body['body_parts'].items():
                        body_keypoints[key] = (value['x'], value['y'])
                    # 将字典的值转换为列表并添加到 keypoints 中
                    # 使用 extend 而不是 append 将坐标展平为两个元素
                    keypoints.append([coord for coords in list(body_keypoints.values()) for coord in coords])
                return keypoints
        return None
        
    def extract_keypoints_dict(self, image_path):
        request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_analysis"
        f = open(image_path, 'rb')
        img = base64.b64encode(f.read())

        params = dict()
        params['image'] = img
        params = urllib.parse.urlencode(params).encode("utf-8")

        access_token = self.get_token()
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded')
        response = urllib.request.urlopen(request)
        content = response.read()

        if content:
            content = content.decode('utf-8')
            data = json.loads(content)
            result = data['person_info']
            keypoints = []
            for body in result:
                body_keypoints = {}
                for key, value in body['body_parts'].items():
                    body_keypoints[key] = (value['x'], value['y'])
                keypoints.append(body_keypoints)
            return keypoints
        return None
