import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from SafetyCheckUtil import MetalImperfectionsUtil
from tensorflow.keras.models import load_model
import tensorflow as tf
from Patches_localizator import Patches_localizator
from Scratch_localizator import Scratch_localizator


class MetalImperfections:
    def __init__(self):
        self.__gpu_setup()
        self.cnn = load_model('./CNN_UTIL/weights_improvement.52-0.0150.h5')

    def recognize(self, path_image):
        '''
        Clasifica la imagen y, a continuación, se llama al método
        __getBndbox__ para obtener una lista de bounding boxes
        :param path_image: Ruta del fichero de la imagen
        :return: Tupla con la predicción del error y la lista de bounding boxes
        '''
        miu = MetalImperfectionsUtil()
        # test_files=miu.read_csv_file('./mi_test.csv')
        #
        # dir_path='./NEU-DET/IMAGES'
        x_test = miu.read_one_image(path_image)

        # Se normalizan los valores
        x_test = x_test.astype('float32')/255.0

        # Se aplica el método predict sobre los ejemplos de test con ruido
        yhat = self.cnn.predict(x_test)

        # print(np.argmax(yhat[0, :]), '<-- yhat')
        label = miu.get_label_text(np.argmax(yhat[0, :]))
        
        bounding_boxes = []
        if label == 'scratches':
            sratch_localizator = Scratch_localizator()
            bounding_boxes = sratch_localizator.localize(path_image)
        elif label == 'patches':
            patches_localizator = Patches_localizator()
            bounding_boxes = patches_localizator.localize(path_image)

        return label, bounding_boxes


    def __getBndbox__(self, image, label):
        '''
        Calcula la lista de boundig boxes de una imagen de un determinado fallo.
        :param image: La imagen
        :param label: Etiqueta clasificada
        :return: Lista de bounding boxes
        '''
        pass

    def __gpu_setup(self):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config=tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.compat.v1.Session(config=config)


if __name__ == '__main__':
    mi = MetalImperfections()
    label, bndbox = mi.recognize('./NEU-DET/IMAGES/scratches_30.jpg')
    print(label)
    label, bndbox = mi.recognize('./NEU-DET/IMAGES/inclusion_1.jpg')
    print(label)
    label, bndbox = mi.recognize('./NEU-DET/IMAGES/scratches_1.jpg')
    print(label)
    label, bndbox=mi.recognize('./NEU-DET/IMAGES/crazing_1.jpg')
    print(label)

