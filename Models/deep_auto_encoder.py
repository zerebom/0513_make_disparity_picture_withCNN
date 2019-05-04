from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Input, MaxPool2D, UpSampling2D, Lambda,Conv2DTranspose

class Deep_auto_encoder:
    def __init__(self,inputs):
        self.model=self.create_model(inputs)

    def create_model(self,inputs):
        conv1 = Conv2D(32, (3, 3), (1, 1), activation='relu', padding='same')(inputs)
        conv2 = Conv2D(64, (3, 3), (2, 2), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(128, (3, 3), (2, 2), activation='relu', padding='same')(conv2)
        conv4 = Conv2D(256, (3, 3), (2, 2), activation='relu', padding='same')(conv3)
        conv5 = Conv2D(512, (3, 3), (2, 2), activation='relu', padding='same')(conv4)
        conv6 = Conv2D(1024, (3, 3), (2, 2), activation='relu', padding='same')(conv5)



        # Decoder
        
        deconv1 = Conv2DTranspose(256, (3, 3), (2, 2), activation='relu', padding='same')(conv6)
        deconv2 = Conv2DTranspose(256, (3, 3), (2, 2), activation='relu', padding='same')(deconv1)
        deconv3 = Conv2DTranspose(256, (3, 3), (2, 2), activation='relu', padding='same')(deconv2)
        deconv4 = Conv2DTranspose(128, (3, 3), (2, 2), activation='relu', padding='same')(deconv3)
        deconv5 = Conv2DTranspose(64, (3, 3), (2, 2), activation='relu', padding='same')(deconv4)
        deconv6 = Conv2DTranspose(32, (3, 3), (2, 2), activation='relu', padding='same')(deconv5)
        output = Conv2D(3, (3, 3), (2, 2), activation='relu', padding='same')(deconv6)
        return Model(inputs,output)


