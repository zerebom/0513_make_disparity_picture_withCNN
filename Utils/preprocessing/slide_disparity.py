
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from PIL import Image
import os
import glob
import re

def slide_disparity(rgb: 'path_str', dis: 'path_str', save_path: 'str', r2l=True) -> 'save_png':
    prefix = "left" if r2l else "right"
    img_name=re.sub(r'.+\\','',os.path.dirname(rgb))
    
    ld = load_img(dis, color_mode="grayscale")
    lr = load_img(rgb)

    r = img_to_array(lr).astype(np.int16)
    d = img_to_array(ld).astype(np.int16)

    #arangeを3次元行列に展開した
    X = np.zeros(r.shape).astype(np.int16)
    value = np.arange(0, r.shape[1]).astype(np.int16)
    #valueにNoneで新しい次元を作成している
    X[:, :, :] = value[None, :, None]
    #3次元行列を2次元行列にしている。
    d = d[:, :, 0]

    #写像先指定行列の作成
    plot_map = np.zeros(d.shape)
    
    #画素をプラス方向に動かす場合(right->left)
    #写像もとに存在しない端っこは消す
    if r2l:
        plot_map[:, :] = np.arange(0, d.shape[1]) + d
        plot_map = np.where(plot_map >= d.shape[1], d.shape[1] - 1, plot_map)
    else:
        plot_map[:, :] = np.arange(0, d.shape[1]) - d
        plot_map = np.where(plot_map <0, 0, plot_map)


    plot_map = plot_map.astype(np.int16)
    #shape[1]->X,shape[0]->Y
    #写像先指定行列をもとに、右目視差画像を作成
    after_array = np.zeros((d.shape[0], d.shape[1], 3))
    for y in range(plot_map.shape[0]):
            after_array[y, plot_map[y,:],:] = r[y,:,:]
    
    after_png = array_to_img(after_array)
    save_dir = os.path.join(save_path, prefix)
    
    os.makedirs(save_dir,exist_ok=True)
    after_png.save(save_dir+'\\'+ img_name+ '.png')


    
if __name__ == '__main__':
    disp1s = glob.glob(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\Original\*\disp1.png')
    disp5s = glob.glob(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\Original\*\disp5.png')
    view1s = glob.glob(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\Original\*\view1.png')
    view5s = glob.glob(r'C:\Users\icech\Desktop\lab2019\2019_4\Data\Original\*\view5.png')

    for rgb,dis in zip(view1s,disp1s):
        slide_disparity(rgb, dis, save_path=r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis', r2l=False)

    for rgb, dis in zip(view5s, disp5s):
        slide_disparity(rgb, dis, save_path=r'C:\Users\icech\Desktop\lab2019\2019_4\Data\slide_dis', r2l=True)
