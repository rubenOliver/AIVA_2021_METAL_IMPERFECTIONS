# This code has been carried out for the Applications subject of the
# Master's Degree in Computer Vision at the Rey Juan Carlos University
# of Madrid.
# Date: April 2021
# Authors: Rubén Oliver, Ismael Linares and Juan Luis Carrillo

import numpy as np
from SafetyCheckUtil import MetalImperfectionsUtil
from tensorflow.keras.models import load_model
import tensorflow as tf
from Patches_localizator import Patches_localizator
from Scratch_localizator import Scratch_localizator


class RecognizerMetalImperfections:
    '''

    '''
    def __init__(self):
        self.__gpu_setup()

        # Loading the best neural network .h5 model
        self.cnn = load_model('./CNN_UTIL/weights_improvement.52-0.0150.h5')

    def __gpu_setup(self):
        '''
        This method is necessary to properly configure tensorflow.
        Without this call, this library may cause some failure
        :return:
        '''
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config=tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        tf.compat.v1.Session(config=config)

    def recognize(self, path_image):
        '''
        Classify the image and get a list of bounding boxes.
        :param path_image: Path of the image file
        :return: Tuple with error prediction and bounding box list
        '''
        miu = MetalImperfectionsUtil()
        x_test = miu.read_one_image(path_image)

        # Data normalization
        x_test = x_test.astype('float32')/255.0

        # The predict method is applied to the image
        yhat = self.cnn.predict(x_test)

        # Get the label in text format
        label = miu.get_label_text(np.argmax(yhat[0, :]))

        # Get the bounding boxes for scratches and patches classes
        bounding_boxes = []
        if label == 'scratches':
            sratch_localizator = Scratch_localizator()
            bounding_boxes = sratch_localizator.localize(path_image)
        elif label == 'patches':
            patches_localizator = Patches_localizator()
            bounding_boxes = patches_localizator.localize(path_image)

        return label, bounding_boxes




if __name__ == '__main__':
    # Create instances of the class and recognize some examples
    mi = RecognizerMetalImperfections()
    label, bndbox = mi.recognize('./NEU-DET/IMAGES/scratches_30.jpg')
    print(label)
    label, bndbox = mi.recognize('./NEU-DET/IMAGES/inclusion_1.jpg')
    print(label)
    label, bndbox = mi.recognize('./NEU-DET/IMAGES/scratches_1.jpg')
    print(label)
    label, bndbox=mi.recognize('./NEU-DET/IMAGES/crazing_1.jpg')
    print(label)

