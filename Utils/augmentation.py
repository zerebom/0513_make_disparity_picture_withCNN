import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


param = {
    rotation_range = 0,
    width_shift_range = 0,
    height_shift_range = 0,
    shear_range = 0,
    zoom_range = 0,
    horizontal_flip = False,
    vertical_flip = False
}

datagen = ImageDataGenerator(**param)
