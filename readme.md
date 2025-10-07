1）在文件解压后，将文件夹放置在一个没有中文的路径下。
2）本项目程序编写在pycharm下完成，建议查看、测试程序时使用pycharm打开。
在设置中选择项目:pose_detect，并将python解释器设置为/pose_detect/env/python.exe。
3）项目的基本结构及功能如下：
pose_detect	Base_pose 放置关键点提取的相应模型
			checkpoints 关键点提取的机器学习预训练模型
			env 本项目的虚拟环境
			feature_data 提取的特征点数据、训练集、测试集储存在该目录下
			picture 测试照片放置场所
			video 测试视频放置场所
			pose GUI 训练所用图片集储存场所
			common.py 放置规则识别的肢体动作通用函数
			main-hand.py 规则识别手指动作的程序
			main-pose-picture.py 规则识别图片中的肢体动作
			main-pose-video.py 规则识别视频中的肢体动作
			main-pose-camera.py 规则识别摄像头中的肢体动作
			point.py 提取关键点的程序
			train.py 利用提取的关键点进行训练的程序
			detect.py 利用机器学习特征识别图片中的肢体动作
			detect_video.py 利用机器学习特征识别视频中的肢体动作
			detect_camera.py 利用机器学习特征识别摄像头中的肢体动作
4）实际执行时，可以直接双击点击.bat文件进行程序的执行
