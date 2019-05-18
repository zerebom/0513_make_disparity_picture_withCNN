from PIL import Image
import numpy as np
from datetime import datetime as dt
import os
from statistics import mean, median, variance, stdev
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator


class Reporter:
    ROOT_DIR = "Result"
    IMAGE_DIR = "image"
    LEARNING_DIR = "learning"
    INFO_DIR = "info"
    MODEL_DIR = "model"
    PARAMETER = "parameter.txt"
    IMAGE_PREFIX = "epoch_"
    IMAGE_EXTENSION = ".png"
    
    def __init__(self, result_dir=None, parser=None):
        self._root_dir = self.ROOT_DIR
        self.create_dirs()
        self.parameters = list()
    # def make_main_dir(self):

    def add_model_name(self, model_name):
        if not type(model_name) is str:
            raise ValueError('model_name is not str.')

        self.model_name = model_name
    def add_val_loss(self, val_loss):
        self.val_loss = str(round(min(val_loss)))

    def generate_main_dir(self):
        main_dir = self.val_loss + '_' + dt.today().strftime("%Y%m%d_%H%M") + '_' + self.model_name
        self.main_dir = os.path.join(self._root_dir, main_dir)
        os.makedirs(self.main_dir, exist_ok=True)

    def create_dirs(self):
        os.makedirs(self._root_dir, exist_ok=True)

    def plot_history(self,history,title='loss'):
        # 後でfontsize変える
        plt.rcParams['axes.linewidth'] = 1.0  # axis line width
        plt.rcParams["font.size"] = 24  # 全体のフォントサイズが変更されます。
        plt.rcParams['axes.grid'] = True  # make grid
        plt.plot(history.history['loss'], linewidth=1.5, marker='o')
        plt.plot(history.history['val_loss'], linewidth=1., marker='o')
        plt.tick_params(labelsize=20)

        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='upper right', fontsize=18)
        plt.tight_layout()

        plt.savefig(os.path.join(self.main_dir, title + self.IMAGE_EXTENSION))
        if len(history.history['val_loss'])>=10:
            plt.xlim(10, len(history.history['val_loss']))
            plt.ylim(0, int(history.history['val_loss'][9]*1.1))

        plt.savefig(os.path.join(self.main_dir, title +'_remove_outlies_'+ self.IMAGE_EXTENSION))

    def add_log_documents(self, add_message):
        self.parameters.append(add_message)


    
    def save_params(self,parser,history):
        
        #early_stoppingを考慮
        self.parameters.append("Number of epochs:" + str(len(history.history['val_loss'])))
        self.parameters.append("Batch size:" + str(parser.batch_size))
        self.parameters.append("Training rate:" + str(parser.trainrate))
        self.parameters.append("Augmentation:" + str(parser.augmentation))
        self.parameters.append("input_channel:" + str(parser.input_channel))
        self.parameters.append("min_val_loss:" + str(min(history.history['val_loss'])))
        self.parameters.append("min_loss:" + str(min(history.history['loss'])))

        # self.parameters.append("L2 regularization:" + str(parser.l2reg))
        output = "\n".join(self.parameters)
        filename=os.path.join(self.main_dir,self.PARAMETER)

        with open(filename, mode='w') as f:
            f.write(output)

    @staticmethod
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def plot_predict(self, img_num_list, Left_RGB, Right_RGB, preds, INPUT_SIZE, max_output=20,save_folder='train'):
        if len(img_num_list) > max_output:
            img_num_list=img_num_list[:max_output]
        for i, num in enumerate(img_num_list):
            if i == 1:
                print(preds[i].astype(np.uint8))
                        
            pred_img = array_to_img(preds[i].astype(np.uint8))
           
            train_img = load_img(Left_RGB[num], target_size=INPUT_SIZE)
            teach_img = load_img(Right_RGB[num], target_size=INPUT_SIZE)
            concat_img = self.get_concat_h(train_img, pred_img)
            concat_img = self.get_concat_h(concat_img, teach_img)
            os.makedirs(os.path.join(self.main_dir,save_folder), exist_ok=True)
            array_to_img(concat_img).save(os.path.join(self.main_dir, save_folder, f'pred_{num}.png'))



