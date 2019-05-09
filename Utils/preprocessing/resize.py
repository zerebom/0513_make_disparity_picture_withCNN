#slide画像用に、path_changesより低機能のコードを作成
import glob
import os
import re
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import numpy as np


def path_changes(picture_names, parent_dic, child_dic, original_pic_paths, init_size=(1024, 1110)):
        #画像の種類ごとにpathを取得する(左目disp->右目disp->...)
        for path, pic_name in zip(original_pic_paths, picture_names):
            image = Image.open(path)
            if init_size:
                #disparitymapのみに使用↓
                image = image.resize(init_size)

            #親ディレクトリ/画像種類別ディレクトリ/画像名.png
            os.makedirs(child_dic, exist_ok=True)
            image.save( child_dic + r'\\' + pic_name)


if __name__ == "__main__":
    L_original_pic_paths=glob.glob(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide\*')
    R_original_pic_paths = glob.glob(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide\*')
    
    parent_dic = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify'
    picture_names = [re.sub(r'.+\\', '', x) for x in L_original_pic_paths]

    child_dic_L=r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide2'
    child_dic_R=r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide2'
        
    path_changes(picture_names, parent_dic, child_dic_L, L_original_pic_paths, init_size=(1024, 1110))
    path_changes(picture_names, parent_dic, child_dic_R, R_original_pic_paths, init_size=(1024, 1110))

    
    


