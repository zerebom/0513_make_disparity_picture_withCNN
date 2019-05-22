import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint

class DN_CNN:
    def __init__(self,input_channel_count,output_channel_count, first_layer_filter_count,parser):
        self.name = self.__class__.__name__.lower()
        self.INPUT_IMAGE_SIZE = 256
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.parser=parser
        # self.CONV_STRIDE = 2
        # self.CONV_PADDING = (1, 1)

        # (128 x 128 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))



        #padding='same'->出力が入力と同じになる。
        x = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same')(inputs)
        x = Activation('relu')(x)
        for i in range(16):
            x=self._add_encoding_layer(first_layer_filter_count, x)
            # x = Conv2D(64, (3, 3), padding='same')(x)
            # X = BatchNormalization()(x)
            # x = Activation('relu')(x)
        
        x = Conv2D(output_channel_count, self.CONV_FILTER_SIZE, padding='same')(x)
        if self.parser.normalize_luminance:
            print()
            x = Activation('tanh')(x)        

        self.DN_CNN = Model(inputs, x)

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = Activation('relu')(sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, padding='same')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def get_model(self):
        return self.DN_CNN
    
    def get_name(self):
        return self.name
