from PIL import Image
import glob
import os
import json
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import numpy as np
import math
import random

"""
np.stackした後でデータを扱うか、
list型で扱うか考える必要がある。
またどの段階で変換するか。
可変性を残しつつ、最後はinput,teachをきれいに返すような形でmain.pyに送りたい。
今のところの課題
・メンバ変数
・inputのconcatnateをいつやるか（main.pyメソッドを呼び出して追加がいいかな？)（しかも、parserからIOできるように)

"""

config = json.load(open('./settings.json'))


class Loader(object):
    # コンストラクタ
    def __init__(self, json_paths, batch_size, init_size=(256, 256)):
        self.size = init_size
        self.DATASET_PATH = json_paths
        self.add_member()
        self.batch_size = batch_size



    # def define_IO(self):
    def add_member(self):
        """
        jsonファイルに記載されている、pathをクラスメンバとして登録する。
        self.Left_RGBとかが追加されている。
        """
        for key, path in self.DATASET_PATH.items():
            setattr(self, key, glob.glob(os.path.join(path, '*png')))

    def concat_left_right(self):
        self.Left_slide += self.Right_slide
        self.Left_RGB += self.Right_RGB
        self.Left_disparity += self.Left_disparity
        self.Right_disparity += self.Right_disparity
        print('Done concat_left_right.')

    # def extract_paths(self,load_dir:'str')->'path_list':
    #     return glob.glob(os.path.join(self.DATASET_PATH[load_dir], '*png'))

    # def add_input(self, batch_list, add_channel):
    #     self.load_batch_img_array(batch_list, add_channel)

    def return_gen(self):
        self.imgs_length = len(self.Left_RGB)
        self.train_paths = (self.Left_slide, self.Right_disparity, self.Left_disparity)
        self.teach_path = self.Left_RGB
        self.train_list, self.valid_list, self.test_list = self.train_valid_test_splits(self.imgs_length)
        self.train_steps = math.ceil(len(self.train_list) / self.batch_size)
        self.valid_steps = math.ceil(len(self.valid_list) / self.batch_size)
        self.test_steps = math.ceil(len(self.test_list) / self.batch_size)
        self.train_gen = self.generator_with_preprocessing(
            self.train_list, self.batch_size, self.teach_path, self.train_paths)
        self.valid_gen = self.generator_with_preprocessing(
            self.valid_list, self.batch_size, self.teach_path, self.train_paths)
        self.test_gen = self.generator_with_preprocessing(
            self.test_list, self.batch_size, self.teach_path, self.train_paths)
        return self.train_gen, self.valid_gen, self.test_gen

    def return_step(self):
        return self.train_steps, self.valid_steps, self.test_steps

    @staticmethod
    def train_valid_test_splits(imgs_length: 'int', train_rate=0.8, valid_rate=0.1, test_rate=0.1):
        data_array = list(range(imgs_length))
        tr = math.floor(imgs_length * train_rate)
        vl = math.floor(imgs_length * (train_rate + valid_rate))

        random.shuffle(data_array)
        train_list = data_array[:tr]
        valid_list = data_array[tr:vl]
        test_list = data_array[vl:]

        return train_list, valid_list, test_list

    def load_batch_img_array(self, batch_list, prepro_callback=False):
        teach_img_list = []
        input_img_list = []
        for i in batch_list:
            LS_img = img_to_array(
                load_img(self.Left_slide[i], color_mode='rgb', target_size=self.size)).astype(np.uint8)
            LD_img = img_to_array(
                load_img(self.Left_disparity[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)
            RD_img = img_to_array(
                load_img(self.Right_disparity[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)

            input_img = np.concatenate((LS_img, LD_img, RD_img), 2).astype(np.uint8)
            teach_img = img_to_array(
                load_img(self.Left_RGB[i], color_mode='rgb', target_size=self.size)).astype(np.uint8)
            
            #バッチサイズが二倍になってしまう                
            # if add_Right:
            #     RS_img = img_to_array(
            #     load_img(self.Right_slide[i], color_mode='rgb', target_size=self.size)).astype(np.uint8)
            #     LD_img = img_to_array(
            #     load_img(self.Left_disparity[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)
            #     RD_img = img_to_array(
            #     load_img(self.Right_disparity[i], color_mode='grayscale', target_size=self.size)).astype(np.uint8)
                
            #     input_img2 = np.concatenate((RS_img, LD_img, RD_img), 2).astype(np.uint8)
            #     teach_img2 = img_to_array(load_img(self.Right_RGB[i], color_mode='rgb', target_size=self.size)).astype(np.uint8)


            input_img_list.append(input_img)
            teach_img_list.append(teach_img)

        return np.stack(input_img_list), np.stack(teach_img_list)

    def generator_with_preprocessing(self, img_id_list, batch_size, teach_path, *input_paths):
        while True:

            # if shuffle:
            #     np.random.shuffle(img_id_list)

            # tupleを剥いてる
            input_paths = input_paths[0]
            for i in range(0, len(img_id_list), batch_size):
                batch_list = img_id_list[i:i + batch_size]

                # print(batch_input)
                batch_input, batch_teach = self.load_batch_img_array(batch_list)

                yield(batch_input, batch_teach)


class DataSet:
    pass
    # @staticmethod
    # def concat_channel(base_arrays, add_arrays,:
    #     img_list = []
    #     if len(base_arrays) != len(add_arrays):
    #         raise ValueError("concat imgs must be same size.")

    #     for base, add in zip(base_arrays, add_arrays):
    #         img = np.concatenate((base, add), 2).astype(np.uint8)
    #         img_list.append(img)

    #     return np.stack(img_list)
