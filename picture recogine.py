#根据训练好的模型进行人脸的识别，是从文件夹中
#-*-coding:utf8-*-#
import cv2
from datetime import datetime
from face_train import Model
from load_dataset import resize_image
import os
import numpy as np
IMAGE_SIZE = 64
if __name__ == '__main__':
    #对图片进行统一格式的处理
    def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
        top, bottom, left, right = (0, 0, 0, 0)

        # 获取图像尺寸
        h, w, _ = image.shape
         # 对于长宽不相等的图片，找到最长的一边
        longest_edge = max(h, w)

        # 计算短边需要增加多上像素宽度使其与长边等长
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass

            # RGB颜色
        BLACK = [0, 0, 0]

        # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
        constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

        # 调整图像大小并返回
        return cv2.resize(constant, (height, width))


    # 函数的作用是根据我们输入的路径进行图片的处理，统一的格式
    def read_path(path_name):
        for dir_item in os.listdir(path_name):
            # 从初始路径开始叠加，合并成可识别的操作路径
            full_path = os.path.abspath(os.path.join(path_name, dir_item))
            if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
                read_path(full_path)
            else:  # 文件
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path)
                    image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)  # 从我们的文件夹中读取图片然后进行统一规格的处理
        return image


    # 从指定路径读取训练数据
    def load_data(path_name):
        images = read_path(path_name)
        # 将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
        # 图片，IMAGE_SIZE为64，故对我来说尺寸为1000 * 64 * 64 * 3
        # 图片为64 * 64像素,一个像素3个颜色值(RGB)
        images = np.array(images)
        return images

    # 加载模型，训练好模型。
    model = Model()
    model.load_model(file_path='./lbo.face.model.h5')

    def detectFaces(image_path):
        img = read_path(image_path)  # 读取读片
        # 加载人脸识别的文件
        face_cascade = cv2.CascadeClassifier("F:\\Anaconda\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")

        # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
                # 对灰度图片进行检测,返回的值是一个列表，里面有四个元素
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))  # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
        if len(faces) > 0:
            for faceRect in faces:
                x, y, w, h = faceRect
                images = gray[y - 10: y + h + 10, x - 10: x + w + 10]
                images = load_data('D:\\face\\sun')
                faceID = model.face_predict(images)
                if faceID == 1:
                    print("识别出图片中的人物是孙顺利")
                elif faceID==2:
                    print("识别出图片中的人物是靳书宝")
                else:
                    print("识别出图片中的人物是王立波")
    time1=datetime.now()
    detectFaces('D:\\face\\sun')

    time2=datetime.now()
    print("耗时："+str(time2-time1))


