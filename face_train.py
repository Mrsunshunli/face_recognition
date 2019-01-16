#coding=gbk
import random

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from load_dataset import load_data, resize_image, IMAGE_SIZE


class Dataset:
    def __init__(self, path_name):
        # ѵ����
        self.train_images = None
        self.train_labels = None

        # ��֤��
        self.valid_images = None
        self.valid_labels = None

        # ���Լ�
        self.test_images = None
        self.test_labels = None

        # ���ݼ�����·��
        self.path_name = path_name

        # ��ǰ����õ�ά��˳��
        self.input_shape = None

    # �������ݼ������ս�����֤��ԭ�򻮷����ݼ����������Ԥ������
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=3):
        # �������ݼ����ڴ�
        images, labels = load_data(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # ��ǰ��ά��˳�����Ϊ'th'��������ͼƬ����ʱ��˳��Ϊ��channels,rows,cols������:rows,cols,channels
        # �ⲿ�ִ�����Ǹ���keras��Ҫ���ά��˳������ѵ�����ݼ�
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # ���ѵ��������֤�������Լ�������
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            # ���ǵ�ģ��ʹ��categorical_crossentropy��Ϊ��ʧ�����������Ҫ�����������nb_classes��
            # ����ǩ����one-hot����ʹ�������������������ǵ����ֻ�����֣�����ת�����ǩ���ݱ�Ϊ��ά
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            # �������ݸ��㻯�Ա��һ��
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            # �����һ��,ͼ��ĸ�����ֵ��һ����0~1����
            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels


# CNN����ģ����
class Model:
    def __init__(self):
        self.model = None

        # ����ģ��

    def build_model(self, dataset, nb_classes=3):
        # ����һ���յ�����ģ�ͣ�����һ�����Զѵ�ģ�ͣ����������ᱻ˳����ӣ�רҵ����Ϊ���ģ�ͻ����Զѵ�ģ��
        self.model = Sequential()

        # ���´��뽫˳�����CNN������Ҫ�ĸ��㣬һ��add����һ�������
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=dataset.input_shape))  # 1 2ά�����
        self.model.add(Activation('relu'))  # 2 �������

        self.model.add(Convolution2D(32, 3, 3))  # 3 2ά�����
        self.model.add(Activation('relu'))  # 4 �������

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 5 �ػ���
        self.model.add(Dropout(0.25))  # 6 Dropout��

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))  # 7  2ά�����
        self.model.add(Activation('relu'))  # 8  �������

        self.model.add(Convolution2D(64, 3, 3))  # 9  2ά�����
        self.model.add(Activation('relu'))  # 10 �������

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 11 �ػ���
        self.model.add(Dropout(0.25))  # 12 Dropout��

        self.model.add(Flatten())  # 13 Flatten��
        self.model.add(Dense(512))  # 14 Dense��,�ֱ�����ȫ���Ӳ�
        self.model.add(Activation('relu'))  # 15 �������
        self.model.add(Dropout(0.5))  # 16 Dropout��
        self.model.add(Dense(nb_classes))  # 17 Dense��
        self.model.add(Activation('softmax'))  # 18 ����㣬������ս��

        # ���ģ�͸ſ�
        self.model.summary()

    # ѵ��ģ��
    def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.001, decay=1e-6,
                  momentum=0.9, nesterov=True)  # ����SGD+momentum���Ż�������ѵ������������һ���Ż�������
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # ���ʵ�ʵ�ģ�����ù���

        # ��ʹ��������������ν���������Ǵ������ṩ��ѵ��������������ת����ת���������ȷ��������µ�
        # ѵ�����ݣ�����ʶ������ѵ�����ݹ�ģ������ģ��ѵ����
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # ʹ��ʵʱ��������
        else:
            # �����������������������������䷵��һ������������datagen��datagenÿ������һ
            # ��������һ�����ݣ�˳�����ɣ�����ʡ�ڴ棬��ʵ����python������������
            datagen = ImageDataGenerator(
                featurewise_center=False,  # �Ƿ�ʹ��������ȥ���Ļ�����ֵΪ0����
                samplewise_center=False,  # �Ƿ�ʹ�������ݵ�ÿ��������ֵΪ0
                featurewise_std_normalization=False,  # �Ƿ����ݱ�׼�����������ݳ������ݼ��ı�׼�
                samplewise_std_normalization=False,  # �Ƿ�ÿ���������ݳ�������ı�׼��
                zca_whitening=False,  # �Ƿ����������ʩ��ZCA�׻�
                rotation_range=20,  # ��������ʱͼƬ���ת���ĽǶ�(��ΧΪ0��180)
                width_shift_range=0.2,  # ��������ʱͼƬˮƽƫ�Ƶķ��ȣ���λΪͼƬ��ȵ�ռ�ȣ�0~1֮��ĸ�������
                height_shift_range=0.2,  # ͬ�ϣ�ֻ���������Ǵ�ֱ
                horizontal_flip=True,  # �Ƿ�������ˮƽ��ת
                vertical_flip=False)  # �Ƿ���������ֱ��ת

            # ��������ѵ������������������������ֵ��һ����ZCA�׻��ȴ���
            datagen.fit(dataset.train_images)

            # ������������ʼѵ��ģ��
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))

    MODEL_PATH = './lbo.face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # ʶ������
    def face_predict(self, image):
        # ��Ȼ�Ǹ��ݺ��ϵͳȷ��ά��˳��
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # �ߴ������ѵ����һ�¶�Ӧ����IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # ��ģ��ѵ����ͬ�����ֻ�����1��ͼƬ����Ԥ��
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # ���㲢��һ��
        image = image.astype('float32')
        image /= 255

        # �����������ڸ������ĸ��ʣ������Ƕ�ֵ�����ú������������ͼ������0��1�ĸ��ʸ�Ϊ����
        result = self.model.predict_proba(image)
        print('result:', result)

        # �������Ԥ�⣺0����1
        result = self.model.predict_classes(image)

        # �������Ԥ����
        return result[0]


if __name__ == '__main__':
    dataset = Dataset('./picture/')
    dataset.load()

    model = Model()
    model.build_model(dataset)

    # ��ǰ��ӵĲ���build_model()�����Ĵ���
    model.build_model(dataset)

    # ����ѵ�������Ĵ���
    model.train(dataset)

if __name__ == '__main__':
    dataset = Dataset('./picture/')
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path='./lbo.face.model.h5')

if __name__ == '__main__':
    dataset = Dataset('./picture/')
    dataset.load()

    # ����ģ��
    model = Model()
    model.load_model(file_path='./lbo.face.model.h5')
    model.evaluate(dataset)




