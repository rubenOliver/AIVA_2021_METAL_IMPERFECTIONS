import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class MetalImperfections:

    def recognize(self, path_image):
        '''
        Clasifica la imagen y a continuación se llama al método
        __getBndbox__ para obtener una lista de bounding boxes
        :param path_image: Ruta del fichero de la imagen
        :return: Tupla con la predicción del error y la lista de bounding boxes
        '''
        return 'label', []


    def __getBndbox__(self, image, label):
        '''
        Calcula la lista de boundig boxes de una imagen de un determinado fallo.
        :param image: La imagen
        :param label: Etiqueta clasificada
        :return: Lista de bounding boxes
        '''
        pass

if __name__ == '__main__':
    pass