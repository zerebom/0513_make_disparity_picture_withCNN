# from keras import backend as K
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
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import Models
from tensorflow.python.keras.layers import Input

import tensorflow as tf
from tensorflow import keras

import json
import glob

# INPUT_SIZE = None

config = json.load(open('./settings.json'))
DATASET_PATH = config['dataset_path']

Left_RGB = glob.glob(os.path.join(DATASET_PATH['Left_RGB'], '*png'))
Right_disparity = glob.glob(os.path.join(DATASET_PATH['Right_disparity'], '*png'))
Left_disparity = glob.glob(os.path.join(DATASET_PATH['Left_disparity'], '*png'))

Right_RGB = glob.glob(os.path.join(DATASET_PATH['Right_RGB'], '*png'))

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


def get_5channel_img_and_teach_img_from_img_id_list(batch_list, Left_RGB=Left_RGB, Right_RGB=Right_RGB, Left_disparity=Left_disparity,
                                                  Right_disparity=Right_disparity, INPUT_SIZE=(128, 128)):
    teach_img_list = []
    input_5_channel_img_list = []
    for i in batch_list:
        L_RGB = img_to_array(load_img(Left_RGB[i], target_size=INPUT_SIZE)).astype(np.uint8)
        L_DIS = img_to_array(load_img(Left_disparity[i], grayscale=True, target_size=INPUT_SIZE)).astype(np.uint8)
        R_DIS = img_to_array(load_img(Right_disparity[i], grayscale=True, target_size=INPUT_SIZE)).astype(np.uint8)

        L_RGB=L_RGB/255
        L_DIS=L_DIS/255
        R_DIS=R_DIS/255

        input_5_channel_img = np.concatenate((L_RGB, L_DIS, R_DIS), 2).astype(np.uint8)
        input_5_channel_img_list.append(input_5_channel_img)

        teach_img = img_to_array(load_img(Right_RGB[i], target_size=INPUT_SIZE)).astype(np.uint8)
        teach_img=teach_img/255
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

# ---------------------------model----------------------------------


inputs = Input(shape=(128,128,5), dtype='float')

model = Models.simple_auto_encoder(inputs)
model.compile(optimizer='adam', loss='mse')

model.summary()


# ---------------------------training----------------------------------

batch_size = 4
train_list, valid_list, test_list = train_valid_test_splits(len(Left_RGB))

train_gen = generator_with_preprocessing(train_list, batch_size)
valid_gen = generator_with_preprocessing(valid_list, batch_size)
test_gen = generator_with_preprocessing(test_list, batch_size)

epochs = 300
train_steps = math.ceil(len(train_list) / batch_size)
valid_steps = math.ceil(len(valid_list) / batch_size)
test_steps = math.ceil(len(test_list) / batch_size)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
# keras.backend.set_session(sess)
# config = tf.ConfigProto()

# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8

# sess = tf.InteractiveSession(config=config)

print("start training.")
model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=valid_gen,
    validation_steps=valid_steps)

print("finish training. And start making predict.")

preds = model.predict_generator(test_gen, steps=test_steps, verbose=1)


print("finish making predict. And render preds.")
for i in range(10):
    display_png(array_to_img(preds[i]))
