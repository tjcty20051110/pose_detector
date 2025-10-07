# 导入BaseML的分类模块
from BaseML import Classification as cls

# 可自行调整评分最高的算法
model = cls('KNN')  # K近临分类
model.load_tab_data('feature_data/mediapipe_pose_train.csv')  # 载入训练数据
model.train()  # 模型训练
print("KNN", end='')
model.valid('feature_data/mediapipe_pose_val.csv', metrics='acc')  # 模型评估
model.save('checkpoints/mediapipe_pose_KNN_mode.pkl')

model = cls('SVM')  # 支持向量机分类
model.load_tab_data('feature_data/mediapipe_pose_train.csv')  # 载入训练数据
model.train()
print("SVM", end='')
model.valid('feature_data/mediapipe_pose_val.csv', metrics='acc')
model.save('checkpoints/mediapipe_pose_SVM_mode.pkl')

model = cls('NaiveBayes')  # 朴素贝叶斯分类
model.load_tab_data('feature_data/mediapipe_pose_train.csv')  # 载入训练数据
model.train()
print("NaiveBayes", end='')
model.valid('feature_data/mediapipe_pose_val.csv', metrics='acc')
model.save('checkpoints/mediapipe_pose_NaiveBayes_mode.pkl')

model = cls('CART')  # 决策树分类
model.load_tab_data('feature_data/mediapipe_pose_train.csv')  # 载入训练数据
model.train()
print("CART", end='')
model.valid('feature_data/mediapipe_pose_val.csv', metrics='acc')
model.save('checkpoints/mediapipe_pose_CART_mode.pkl')

model = cls('AdaBoost')  # 自适应增强分类
model.load_tab_data('feature_data/mediapipe_pose_train.csv')  # 载入训练数据
model.train()
print("AdaBoost", end='')
model.valid('feature_data/mediapipe_pose_val.csv', metrics='acc')
model.save('checkpoints/mediapipe_pose_AdaBoost_mode.pkl')

model = cls('MLP')  # 多层感知机分类
model.load_tab_data('feature_data/mediapipe_pose_train.csv')  # 载入训练数据
model.train()
print("MLP", end='')
model.valid('feature_data/mediapipe_pose_val.csv', metrics='acc')
model.save('checkpoints/mediapipe_pose_MLP_mode.pkl')

model = cls('RandomForest')  # 随机森林分类
model.load_tab_data('feature_data/mediapipe_pose_train.csv')  # 载入训练数据
model.train()
print("RandomForest", end='')
model.valid('feature_data/mediapipe_pose_val.csv', metrics='acc')
model.save('checkpoints/mediapipe_pose_RandomForest_mode.pkl')

