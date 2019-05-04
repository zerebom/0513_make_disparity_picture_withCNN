from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Input, MaxPool2D, UpSampling2D, Lambda,Conv2DTranspose

class Simple_auto_encoder(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 128
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        inputs = Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count),dtype='float')

        conv1 = Conv2D(first_layer_filter_count,(3, 3), (1, 1), activation='relu', padding='same')(inputs)
        conv2 = Conv2D(first_layer_filter_count*2,(3, 3), (2, 2), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(first_layer_filter_count*4, (3, 3), (2, 2), activation='relu', padding='same')(conv2)
        conv4 = Conv2D(first_layer_filter_count*8, (3, 3), (2, 2), activation='relu', padding='same')(conv3)

        # Decoder
        deconv1 = Conv2DTranspose(first_layer_filter_count * 8, (3, 3), (2, 2),activation='relu', padding='same')(conv4)
        deconv2 = Conv2DTranspose(first_layer_filter_count * 4, (3, 3), (2, 2),activation='relu', padding='same')(deconv1)
        deconv3 = Conv2DTranspose(first_layer_filter_count * 2, (3, 3), (2, 2),activation='relu', padding='same')(deconv2)
        deconv4 = Conv2DTranspose(first_layer_filter_count, (3, 3), (2, 2),activation='relu', padding='same')(deconv3)
        output = Conv2D(output_channel_count,(3, 3), (2, 2), activation='relu', padding='same')(deconv4)

        self.Simple_auto_encoder=Model(inputs,output)


    def get_model(self):    
        return self.Simple_auto_encoder


