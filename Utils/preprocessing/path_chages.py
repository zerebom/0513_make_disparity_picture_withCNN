import glob
import os
import re
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import numpy as np
"""
予めModifyディレクトリは作っておくこと。
modify_disparity_grayscaleを使うときはRGB部分をコメントアウトしておく。

"""
#disparitymapの輝度値を圧縮率(返還後のX/変換前のX)で割る。
def modify_disparity_grayscale(im, after_width=1024):
    per = im.width / after_width
    im_ar = img_to_array(im)
    im_ar = im_ar / per
    im_ar=im_ar.astype(np.uint8)
    im = array_to_img(im_ar)
    return im


def path_changes(picture_names,parent_dic,child_dic, paths, init_size=(1024, 1110)):
        #画像の種類ごとにpathを取得する(左目disp->右目disp->...)
        for path, pic_name in zip(paths,picture_names):
            image = Image.open(path)         
            if init_size:
                #disparitymapのみに使用↓
                image=modify_disparity_grayscale(image, after_width=1024)
                image = image.resize(init_size)

            #親ディレクトリ/画像種類別ディレクトリ/画像名.png
            os.makedirs(parent_dic + r'\\' + child_dic, exist_ok=True)
            image.save(parent_dic + r'\\' + child_dic + r'\\' + pic_name + '.png')


if __name__ == '__main__':
    #disp1,5とか含まれている、画像の名前込みのディレクトリ群。
    original = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\Original'
    modify = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify'
    
    #ディレクトリ名が画像名になっているので、それを抽出(ex:baby1等)
    picture_paths = glob.glob(original+r'\*')
    picture_names = [re.sub(r'.+\\', '', x) for x in picture_paths]

    path_dict = {
        'Left_disparity': r'\*\disp1.png',
        # 'Left_RGB': r'\*\view1.png',
        'Right_disparity': r'\*\disp5.png',
        # 'Right_RGB': r'\*\view5.png'
        }
    

    for child_dic, path_end in path_dict.items():
        paths = glob.glob(original + path_end)
        #画像保存の親ディレクトリと、その中のフォルダ名を抽出。
        path_changes(modify,picture_names, child_dic, paths, init_size=(1024, 1110))
