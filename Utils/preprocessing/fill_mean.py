import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from PIL import Image
import json
import os
import math
import random
import pandas as pd
import glob
import re

"""
画像を埋める
"""
def fill_mean(im):
    r = img_to_array(im).astype(np.int16)

    R = r[:, :, 0]
    G = r[:, :, 1]
    B = r[:, :, 2]

    Rmean = np.mean(R[R > 0]).astype(np.uint8)
    Gmean = np.mean(G[G > 0]).astype(np.uint8)
    Bmean = np.mean(B[B > 0]).astype(np.uint8)

    fill_r = np.where(r == [0, 0, 0], [Rmean, Gmean, Bmean], r)

    return array_to_img(fill_r)


if __name__ == '__main__':
    Left_slide = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide\splits2'
    Right_slide = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide\splits2'

    Left_mean = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide\splits2_fill'
    Right_mean = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide\splits2_fill'

    Lpaths = glob.glob(Left_slide + r'\*')
    Rpaths = glob.glob(Right_slide + r'\*')

    for Lpath, Rpath in zip(Lpaths, Rpaths):
        l_name = re.sub(r'.+\\', '', Lpath)
        r_name = re.sub(r'.+\\', '', Rpath)
        
        l_im = load_img(Lpath)
        r_im = load_img(Rpath)

        l_im = fill_mean(l_im)
        r_im = fill_mean(r_im)

        os.makedirs(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide\splits2_fill', exist_ok=True)
        os.makedirs(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide\splits2_fill', exist_ok=True)
        
        l_im.save(os.path.join(Left_mean, l_name))
        r_im.save(os.path.join(Right_mean, r_name))


