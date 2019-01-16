#��δ���������Ƕ�����ץȡ����Ƭ����Ԥ����
# coding=gbk
import os
import sys
import numpy as np
import cv2
IMAGE_SIZE = 64

# ����ָ��ͼ���С�����ߴ�,�����е�ͼƬ����64*64
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right =( 0, 0, 0, 0 )

    # ��ȡͼ��ߴ�
    h, w, _ = image.shape

    # ���ڳ�����ȵ�ͼƬ���ҵ����һ��
    longest_edge = max(h, w)

    # ����̱���Ҫ���Ӷ������ؿ��ʹ���볤�ߵȳ�
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

        # RGB��ɫ
    BLACK = [0, 0, 0]

    # ��ͼ�����ӱ߽磬��ͼƬ������ȳ���cv2.BORDER_CONSTANTָ���߽���ɫ��valueָ��
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # ����ͼ���С������
    return cv2.resize(constant, (height, width))


# ��ȡѵ������
images = []#images����ŵ������ǽ��д����ͼƬ��64*64�Ĵ�С
labels = []

#�����������Ǹ������������·������ͼƬ�Ĵ���ͳһ�ĸ�ʽ
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        # �ӳ�ʼ·����ʼ���ӣ��ϲ��ɿ�ʶ��Ĳ���·��
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # ������ļ��У������ݹ����
            read_path(full_path)
        else:  # �ļ�
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)#�����ǵ��ļ����ж�ȡͼƬȻ�����ͳһ���Ĵ���

                # �ſ�������룬���Կ���resize_image()������ʵ�ʵ���Ч��
                cv2.imwrite('1.jpg', image)

                images.append(image)
                labels.append(path_name)

    return images, labels


# ��ָ��·����ȡѵ������
def load_data(path_name):
    images, labels = read_path(path_name)
    i=1
    # �����������ͼƬת����ά���飬�ߴ�Ϊ(ͼƬ����*IMAGE_SIZE*IMAGE_SIZE*3)
    # ͼƬ��IMAGE_SIZEΪ64���ʶ�����˵�ߴ�Ϊ1000 * 64 * 64 * 3
    # ͼƬΪ64 * 64����,һ������3����ɫֵ(RGB)
    images = np.array(images)
    print(images.shape)
    # ��ע���ݣ�'sunshunli'�ļ����¶����ҵ�����ͼ��ȫ��ָ��Ϊ0������һ���ļ�������ͬѧ�ģ�ȫ��ָ��Ϊ1

    # labels = np.array([0 if label.endswith('sunshunli') else i for label in labels])#����ʱ�Ĵ��ǩ�Ĵ���


    ##################################################################################
    #���˶��ǵĴ��ǩ��Ĵ��룬Ҳ�����Ƕ��˵Ĵ���
    slist = []
    print(labels)
    for label in labels:
        if label.endswith('sunshunli'):
            slist.append(0)
        elif label.endswith('jinshubao'):
            slist.append(1)
        else:
            slist.append(2)
    labels=slist

    ###############################################################################################

    return images, labels


if __name__ == '__main__':
    if len(sys.argv) != 1:
            print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        images, labels = load_data('D:\\face\\picture')
