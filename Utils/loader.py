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
    #コンストラクタ
    def __init__(self,json_paths,batch_size, init_size=(256, 256)):
        self.size = init_size
        self.DATASET_PATH = json_paths
        self.add_member()
        self.batch_size = batch_size
        self.imgs_length = len(self.Left_RGB)
        
        
        self.tr_list, self.val_list, self.tes_list = self.tr_val_tes_splits(imgs_length)
        
        self.tr_gen = self.generator_with_preprocessing(self.tr_list, self.batch_size)
        self.val_gen = self.generator_with_preprocessing(self.val_list, self.batch_size)
        self.tes_gen = self.generator_with_preprocessing(self.tes_list, self.batch_size)

        self.tr_steps = math.ceil(len(self.tr_list) / self.batch_size)
        self.val_steps = math.ceil(len(self.val_list) / self.batch_size)
        self.tes_steps = math.ceil(len(self.tes_list) / self.batch_size)

    
    def add_member(self):
        """
        jsonファイルに記載されている、pathをクラスメンバとして登録する。
        """
        for key, path in self.DATASET_PATH.items():
            setattr(self, key, glob.glob(os.path.join(path, '*png')))
             
    # def extract_paths(self,load_dir:'str')->'path_list':
    #     return glob.glob(os.path.join(self.DATASET_PATH[load_dir], '*png'))
    
    def return_array(self, **kwargs):
        




    @staticmethod
    def tr_val_tes_splits(imgs_length: 'int', train_rate=0.8, valid_rate=0.1, test_rate=0.1):
        data_array = list(range(imgs_length))
        tr = math.floor(imgs_length * train_rate)
        vl = math.floor(imgs_length * (train_rate + valid_rate))

        random.shuffle(data_array)
        train_list = data_array[:tr]
        valid_list = data_array[tr:vl]
        test_list = data_array[vl:]

        return train_list, valid_list, test_list
    
    def load_batch_img_array(self, batch_list, paths, input_size,return_4dim=True,prepro_callback=False):
        img_list = []
        for i in batch_list:
            img = img_to_array(load_img(paths[i], target_size=input_size)).astype(np.uint8)
            
            if prepro_callback:
                prepro_callback(img)
            
            img_list.append(img)
        
        if return_4dim:
            return np.stack(img_list)
        else:
            return img_list
    
    @staticmethod
    def concat_channel(base_arrays, add_arrays, return_4dim=True):
        img_list = []
        if len(base_arrays) != len(add_arrays):
            raise ValueError("concat imgs must be same size.")
        
        for base, add in zip(base_arrays, add_arrays):
            img = np.concatenate((base, add), 2).astype(np.uint8)
            img_list.append(img)
        
        if return_4dim:
            return np.stack(img_list)
        else:
            return img_list
    
    def generator_with_preprocessing(self,img_id_list, batch_size, shuffle=False):
        while True:
            if shuffle:
                np.random.shuffle(img_id_list)
            
            for i in range(0, len(img_id_list), batch_size):
                batch_list = img_id_list[i:i + batch_size]
                batch_input = self.load_batch_img_array(batch_list,img_id_list)
                batch_teach = self.load_batch_img_array(batch_list,img_id_list)

                yield(batch_input, batch_teach)

        


class DataSet:
    pass
