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


def bin_im(im):
    im = im.convert('L')
    r = img_to_array(im).astype(np.int16)
    _bin = np.where(r > 0, 255, r)

    return array_to_img(_bin)


if __name__ == '__main__':
    Left_slide = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide\splits2'
    Right_slide = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide\splits2'

    Left_bin = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide\splits2_bin'
    Right_bin = r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide\splits2_bin'

    Lpaths = glob.glob(Left_slide + r'\*')
    Rpaths = glob.glob(Right_slide + r'\*')

    for Lpath, Rpath in zip(Lpaths, Rpaths):
        l_name = re.sub(r'.+\\', '', Lpath)
        r_name = re.sub(r'.+\\', '', Rpath)

        l_im = load_img(Lpath)
        r_im = load_img(Rpath)

        l_im = bin_im(l_im)
        r_im = bin_im(r_im)

        os.makedirs(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Left_slide\splits2_bin', exist_ok=True)
        os.makedirs(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis\Modify\Right_slide\splits2_bin', exist_ok=True)

        l_im.save(os.path.join(Left_bin, l_name))
        r_im.save(os.path.join(Right_bin, r_name))
