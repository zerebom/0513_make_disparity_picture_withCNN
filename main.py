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
from datetime import datetime as dt
from tensorflow import keras
from Utils.reporter import Reporter
from Utils.loader import DataSet, Loader

import json
import glob

INPUT_SIZE = (256, 256)
CONCAT_LEFT_RIGHT=True
CHANGE_SLIDE2_FILL = True


def train(parser):

    configs = json.load(open('./settings.json'))
    reporter = Reporter(parser=parser)
    loader = Loader(configs['dataset_path2'], parser.batch_size)
    
    if CHANGE_SLIDE2_FILL:
        loader.change_slide2fill()
        reporter.add_log_documents('Done change_slide2fill.')

    if CONCAT_LEFT_RIGHT:
        loader.concat_left_right()
        reporter.add_log_documents('Done concat_left_right.')


    train_gen, valid_gen, test_gen = loader.return_gen()
    train_steps, valid_steps, test_steps = loader.return_step()

    # ---------------------------model----------------------------------

    input_channel_count = parser.input_channel
    output_channel_count = 3
    first_layer_filter_count = 32

    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # ---------------------------training----------------------------------
    batch_size = parser.batch_size
    epochs = parser.epoch

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # fit_generatorのコールバック関数の指定・TensorBoardとEarlyStoppingの指定

    logdir = os.path.join('./logs', dt.today().strftime("%Y%m%d_%H%M"))
    os.makedirs(logdir, exist_ok=True)
    tb_cb = TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True)

    es_cb = EarlyStopping(monitor='val_loss', patience=parser.early_stopping, verbose=1, mode='auto')

    print("start training.")
    # Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        # use_multiprocessing=True,
        callbacks=[es_cb, tb_cb])

    print("finish training. And start making predict.")

    train_preds = model.predict_generator(train_gen, steps=train_steps, verbose=1)
    valid_preds = model.predict_generator(valid_gen, steps=valid_steps, verbose=1)
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
    reporter.plot_predict(loader.train_list, loader.Left_slide, loader.Left_RGB,
                          train_preds, INPUT_SIZE, save_folder='train')
    reporter.plot_predict(loader.valid_list, loader.Left_slide, loader.Left_RGB,
                          valid_preds, INPUT_SIZE, save_folder='valid')
    reporter.plot_predict(loader.test_list, loader.Left_slide, loader.Left_RGB,
                          test_preds, INPUT_SIZE, save_folder='test')
    model.save("model.h5")


def get_parser():
    parser = argparse.ArgumentParser(
        prog='generate parallax image using U-Net',
        usage='python main.py',
        description='This module　generate parallax image using U-Net.',
        add_help=True
    )

    parser.add_argument('-e', '--epoch', type=int,
                        default=200, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float,
                        default=0.85, help='Training rate')
    parser.add_argument('-es', '--early_stopping', type=int,
                        default=20, help='early_stopping patience')

    parser.add_argument('-i', '--input_channel', type=int,
                        default=7, help='input_channel')

    parser.add_argument('-a', '--augmentation',
                        action='store_true', help='Number of epochs')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
