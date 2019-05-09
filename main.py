import argparse
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
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from Models import deep_auto_encoder
from Models.simple_auto_encoder import Simple_auto_encoder
from Models.unet import UNet
from tensorflow.python.keras.layers import Input
import tensorflow as tf
import datetime
from tensorflow import keras
from Utils.reporter import Reporter

import json
import glob

INPUT_SIZE = (128, 128)

config = json.load(open('./settings.json'))
DATASET_PATH = config['dataset_path2']

Left_RGB = glob.glob(os.path.join(DATASET_PATH['Left_slide'], '*png'))
Right_disparity = glob.glob(os.path.join(DATASET_PATH['Right_disparity'], '*png'))
Left_disparity = glob.glob(os.path.join(DATASET_PATH['Left_disparity'], '*png'))

Right_RGB = glob.glob(os.path.join(DATASET_PATH['Left_RGB'], '*png'))

# ----------------------function----------------------------------------------------



def train_valid_test_splits(img_total_num, train_rate=0.8, valid_rate=0.1, test_rate=0.1):
    data_array = list(range(img_total_num))
    tr = math.floor(img_total_num * train_rate)
    vl = math.floor(img_total_num * (train_rate + valid_rate))

    random.shuffle(data_array)
    train_list = data_array[:tr]
    valid_list = data_array[tr:vl]
    test_list = data_array[vl:]

    return train_list, valid_list, test_list


def get_input_and_teach_img_from_img_id_list(batch_list, Left_RGB=Left_RGB, Right_RGB=Right_RGB, Left_disparity=Left_disparity,
                                             Right_disparity=Right_disparity, input_channel=3, INPUT_SIZE=(128, 128)):
    teach_img_list = []
    input_img_list = []
    for i in batch_list:
        input_img = img_to_array(load_img(Left_RGB[i], target_size=INPUT_SIZE)).astype(np.uint8)
        # input_img = input_img / 255

        if input_channel == 5:
            L_DIS = img_to_array(load_img(Left_disparity[i], color_mode="grayscale", target_size=INPUT_SIZE)).astype(np.uint8)
            R_DIS = img_to_array(load_img(Right_disparity[i],color_mode = "grayscale", target_size=INPUT_SIZE)).astype(np.uint8)
            input_img = np.concatenate((input_img, L_DIS, R_DIS), 2).astype(np.uint8)

        input_img_list.append(input_img)

        teach_img = img_to_array(load_img(Right_RGB[i], target_size=INPUT_SIZE)).astype(np.uint8)
        # teach_img = teach_img / 255

        teach_img_list.append(teach_img)

# 4次元テンソルに変換している
    return np.stack(input_img_list), np.stack(teach_img_list)


def generator_with_preprocessing(img_id_list, batch_size, input_channel, shuffle=False):
    while True:
        if shuffle:
            np.random.shuffle(img_id_list)
        for i in range(0, len(img_id_list), batch_size):
            batch_list = img_id_list[i:i + batch_size]
            batch_input, batch_teach = get_input_and_teach_img_from_img_id_list(batch_list, input_channel=input_channel)

            yield(batch_input, batch_teach)


def train(parser):
    reporter=Reporter(parser=parser)
    # ---------------------------model----------------------------------

    # inputs = Input(shape=(128, 128, 3), dtype='float')
    input_channel_count = parser.input_channel
    output_channel_count = 3
    first_layer_filter_count = 64

    # network = Simple_auto_encoder(input_channel_count, output_channel_count, first_layer_filter_count)
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # ---------------------------training----------------------------------
    batch_size = parser.batchsize
    epochs = parser.epoch

    train_list, valid_list, test_list = train_valid_test_splits(len(Left_RGB))

    train_gen = generator_with_preprocessing(train_list, batch_size, input_channel=parser.input_channel)
    valid_gen = generator_with_preprocessing(valid_list, batch_size, input_channel=parser.input_channel)
    test_gen = generator_with_preprocessing(test_list, batch_size, input_channel=parser.input_channel)

    train_steps = math.ceil(len(train_list) / batch_size)
    valid_steps = math.ceil(len(valid_list) / batch_size)
    test_steps = math.ceil(len(test_list) / batch_size)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # fit_generatorのコールバック関数の指定・TensorBoardとEarlyStoppingの指定
    tb_cb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    es_cb = EarlyStopping(monitor='val_loss', patience=parser.early_stopping, verbose=1, mode='auto')

    print("start training.")
    # Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
    history=model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        # use_multiprocessing=True,
        callbacks=[es_cb, tb_cb])
    
    print("finish training. And start making predict.")
    
    # train_preds = model.predict_generator(train_gen, steps=train_steps, verbose=1)
    # valid_preds = model.predict_generator(valid_gen, steps=valid_steps, verbose=1)
    test_preds = model.predict_generator(test_gen, steps=test_steps, verbose=1)

    print("finish making predict. And render preds.")

    # ==========================report====================================
    reporter.add_val_loss(history.history['val_loss'])
    reporter.add_model_name(network.__class__.__name__)
    reporter.generate_main_dir()
    reporter.plot_history(history)
    reporter.save_params(parser, history)
    
    input_img_list = []
    # reporter.plot_predict(train_list, Left_RGB, Right_RGB, train_preds, INPUT_SIZE, save_folder='train')
    # reporter.plot_predict(valid_list, Left_RGB, Right_RGB, valid_preds, INPUT_SIZE,save_folder='valid')
    reporter.plot_predict(test_list, Left_RGB, Right_RGB, test_preds, INPUT_SIZE, save_folder='test')
    model.save("model.h5")



def get_parser():
    parser = argparse.ArgumentParser(
        prog='generate parallax image using U-Net',
        usage='python main.py',
        description='This module　generate parallax image using U-Net.',
        add_help=True
    )

    parser.add_argument('-e', '--epoch', type=int,
                        default=100, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int,
                        default=16, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float,
                        default=0.85, help='Training rate')
    parser.add_argument('-es', '--early_stopping', type=int,
                        default=30, help='early_stopping patience')

    parser.add_argument('-i', '--input_channel', type=int,
                        default=5, help='input_channel')

    parser.add_argument('-a', '--augmentation',
                        action='store_true', help='Number of epochs')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
