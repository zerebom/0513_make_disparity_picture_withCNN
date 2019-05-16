import argparse

# from keras import backend as K
import keras.callbacks
from PIL import Image, ImageOps
from IPython.display import display_png
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python import keras
# import keras.backend.tensorflow_backend as KTF
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from Models import simple_auto_encoder, deep_auto_encoder
from tensorflow.python.keras.layers import Input

import tensorflow as tf
import datetime
from tensorflow import keras
# ### hack tf-keras to appear as top level keras
# import sys
# sys.modules['keras'] = keras
# ### end of hack

import json
import glob

INPUT_SIZE = (128, 128)

config = json.load(open('./settings.json'))
DATASET_PATH = config['dataset_path']

Left_RGB = glob.glob(os.path.join(DATASET_PATH['Left_RGB'], '*png'))
Right_disparity = glob.glob(os.path.join(DATASET_PATH['Right_disparity'], '*png'))
Left_disparity = glob.glob(os.path.join(DATASET_PATH['Left_disparity'], '*png'))

Right_RGB = glob.glob(os.path.join(DATASET_PATH['Right_RGB'], '*png'))

# ----------------------function----------------------------------------------------


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def generate_dir_name():
    return datetime.datetime.today().strftime("%Y%m%d_%H%M")


def train_valid_test_splits(img_total_num, train_rate=0.8, valid_rate=0.1, test_rate=0.1):
    data_array = list(range(img_total_num))
    tr = math.floor(img_total_num * train_rate)
    vl = math.floor(img_total_num * (train_rate + valid_rate))

    random.shuffle(data_array)
    train_list = data_array[:tr]
    valid_list = data_array[tr:vl]
    test_list = data_array[vl:]

    return train_list, valid_list, test_list


def get_5channel_img_and_teach_img_from_img_id_list(batch_list, Left_RGB=Left_RGB, Right_RGB=Right_RGB, Left_disparity=Left_disparity,
                                                    Right_disparity=Right_disparity, INPUT_SIZE=(128, 128)):
    teach_img_list = []
    input_5_channel_img_list = []
    for i in batch_list:
        L_RGB = img_to_array(load_img(Left_RGB[i], target_size=INPUT_SIZE)).astype(np.uint8)
        L_DIS = img_to_array(load_img(Left_disparity[i], grayscale=True, target_size=INPUT_SIZE)).astype(np.uint8)
        R_DIS = img_to_array(load_img(Right_disparity[i], grayscale=True, target_size=INPUT_SIZE)).astype(np.uint8)

        L_RGB = L_RGB / 255
        L_DIS = L_DIS / 255
        R_DIS = R_DIS / 255

        input_5_channel_img = np.concatenate((L_RGB, L_DIS, R_DIS), 2).astype(np.uint8)
        input_5_channel_img_list.append(input_5_channel_img)

        teach_img = img_to_array(load_img(Right_RGB[i], target_size=INPUT_SIZE)).astype(np.uint8)
        teach_img = teach_img / 255

        teach_img_list.append(teach_img)

# 4次元テンソルに変換している
    return np.stack(input_5_channel_img_list), np.stack(teach_img_list)


def generator_with_preprocessing(img_id_list, batch_size, shuffle=False):
    while True:
        if shuffle:
            np.random.shuffle(img_id_list)
        for i in range(0, len(img_id_list), batch_size):
            batch_list = img_id_list[i:i + batch_size]
            batch_5, batch_teach = get_5channel_img_and_teach_img_from_img_id_list(batch_list)

            yield(batch_5, batch_teach)


def train(parser):

    # ---------------------------model----------------------------------

    inputs = Input(shape=(128, 128, 5), dtype='float')
    # model = simple_auto_encoder.Simple_auto_encoder(inputs).model
    model = deep_auto_encoder.Deep_auto_encoder(inputs).model
    model.compile(optimizer='adam', loss='mse')
    model.summary()

# ---------------------------training----------------------------------

    batch_size = parser.batch_size
    epochs = parser.epoch

    train_list, valid_list, test_list = train_valid_test_splits(len(Left_RGB))

    train_gen = generator_with_preprocessing(train_list, batch_size)
    valid_gen = generator_with_preprocessing(valid_list, batch_size)
    test_gen = generator_with_preprocessing(test_list, batch_size)

    train_steps = math.ceil(len(train_list) / batch_size)
    valid_steps = math.ceil(len(valid_list) / batch_size)
    test_steps = math.ceil(len(test_list) / batch_size)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # fit_generatorのコールバック関数の指定・TensorBoardとEarlyStoppingの指定
    tb_cb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    es_cb = EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='auto')

    print("start training.")
    # Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        callbacks=[es_cb, tb_cb])

    print("finish training. And start making predict.")

    preds = model.predict_generator(test_gen, steps=test_steps, verbose=1)

    print("finish making predict. And render preds.")
    local_dir_name = generate_dir_name()
    dir_name = os.path.join('./Result/output', local_dir_name)
    os.makedirs(dir_name)

    # ==========================plot predict====================================
    for i, num in enumerate(test_list):
        if i == 1:
            print(preds[i].astype(np.uint8))
        pred_img = array_to_img(preds[i].astype(np.uint8))
        teach_img = load_img(Right_RGB[num], target_size=INPUT_SIZE)

        concat_img = get_concat_h(pred_img, teach_img)
        array_to_img(concat_img).save(os.path.join(dir_name, f'pred_{num}.png'))

    model.save("model.h5")
    # KTF.set_session(old_session)

# =================================parser=====================================


def get_parser():
    parser = argparse.ArgumentParser(
        prog='generate parallax image using U-Net',
        usage='python main.py',
        description='This module　generate parallax image using U-Net.',
        add_help=True
    )

    parser.add_argument('-e', '--epoch', type=int,
                        default=100, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=16, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float,
                        default=0.85, help='Training rate')
    parser.add_argument('-es', '--early_stopping', type=float,
                        default=0.85, help='Training rate')

    parser.add_argument('-a', '--augmentation',
                        action='store_true', help='Number of epochs')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
