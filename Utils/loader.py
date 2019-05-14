from PIL import Image
import glob
import os
import json
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator


config = json.load(open('./settings.json'))

class Loader(object):
    #コンストラクタ
    def __init__(self, init_size=(256, 256)):
        self.size = init_size
        self.DATASET_PATH = config['dataset_path2']
        self.add_menba()

    
    def add_member(self):
        """
        jsonファイルに記載されている、pathをクラスメンバとして登録する。
        """
        for key, path in self.DATASET_PATH.items():
            setattr(self, key, glob.glob(os.path.join(path, '*png')))
             
    # def extract_paths(self,load_dir:'str')->'path_list':
    #     return glob.glob(os.path.join(self.DATASET_PATH[load_dir], '*png'))
    
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
    
    def load_batch_img_array(self, batch_list, paths, input_size=self.size,prepro_callback=False):
        img_list = []
        for i in batch_list:
            img = img_to_array(load_img(paths[i], target_size=input_size)).astype(np.uint8)
            
            if prepro_callback:
                prepro_callback(img)
            
            img_list.append(img)
        return img_list
    
    @staticmethod
    def concat_channel(base_arrays, add_arrays):
        img_list = []
        if len(base_arrays) != len(add_arrays):
            raise ValueError("concat imgs must be same size.")
        
        for base, add in zip(base_arrays, add_arrays):
            img = np.concatenate((base, add), 2).astype(np.uint8)
            img_list.append(img)
        return img_list

    def generator_with_preprocessing(img_id_list, batch_size, input_channel, shuffle=False):
        while True:
            if shuffle:
                np.random.shuffle(img_id_list)
            
            for i in range(0, len(img_id_list), batch_size):
                batch_list = img_id_list[i:i + batch_size]
                batch_input = load_batch_img_array(batch_list,)
                batch_teach

                yield(batch_input, batch_teach)

        


class DataSet:
