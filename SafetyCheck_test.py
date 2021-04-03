"""
Fichero con los test unitarios del entrenamiento y del reconocedor
"""

import unittest
from SafetyCheck import MetalImperfections
import os
from xml.dom import minidom
from SafetyCheckUtil import MetalImperfectionsUtil
import glob
import numpy as np
from Patches_localizator import Patches_localizator
from Scratch_localizator import Scratch_localizator
import cv2 as cv


def __find_index__(target, array):
    for i,row in enumerate(array):
        for j,value in enumerate(row):
            if value == target:
                return (i, j)

def __get_intersection__(bb_results, bndboxs):
    '''
    Se calculan las intersecciones
    :param bb_results: Los bounding-boxes del detector
    :param bndboxs: Los bounding-boxes del ground-truth
    :return: IOU medio para todas las intersecciones
    '''
    auxInter = []

    # Lista con el IOU de cada bndbox del ground thruth con cada prediccion
    bb_intersections = []
    for bndbox in bndboxs:
        auxInter = []
        for bb_result in bb_results:
            xA = max(bb_result[0], bndbox[0])
            yA = max(bb_result[1], bndbox[1])
            xB = min(bb_result[2], bndbox[2])
            yB = min(bb_result[3], bndbox[3])

            # Area de interseccion entre rectangulos
            interArea = max(0, (xB - xA + 1)) * max(0,(yB - yA + 1))

            # Area para cada bounding box del ground thruth y prediccion
            boxAArea = (bb_result[2] - bb_result[0] + 1) * (bb_result[3] - bb_result[1] + 1)
            boxBArea = (bndbox[2] - bndbox[0] + 1) * (bndbox[3] - bndbox[1] + 1)

            # Calculo de intersection over union para el grado de acierto
            iou = interArea / float(boxAArea + boxBArea - interArea)
            auxInter.append(iou)
        bb_intersections.append(auxInter)
    final_intersections = []
    counter = 0
    gt_len = len(bndboxs)
    for i in range(0, gt_len):
        if not bb_intersections[0]:
            final_intersections.append(0.0)
        else:
            # Maximo IOU de todos los bndbox
            maxValue = np.max(bb_intersections)
            i, j = __find_index__(maxValue, bb_intersections)

            # Se elimina de la lista el bndbox del GT y la columna con el bndbox predicho 
            # para solo tener un correspondencia para cada bndbox del GT
            del bb_intersections[i]
            if bb_intersections:
                for row in bb_intersections:
                        del row[j]
            final_intersections.append(maxValue)
    return sum(final_intersections) / len(final_intersections)

def __get_gt_bndbox__(path_anno):
    '''
    Obtiene los bounding-boxes del ground-truth
    :param path_anno: Ruta de los documentos de anotaciones
    :return: Lista de bounding-boxes del ground-truth
    '''
    bndboxs = []
    doc = minidom.parse(path_anno)
    objects = doc.getElementsByTagName("object")
    for object in objects:
        xmin = object.getElementsByTagName("xmin")[0].firstChild.data
        ymin = object.getElementsByTagName("ymin")[0].firstChild.data
        xmax = object.getElementsByTagName("xmax")[0].firstChild.data
        ymax = object.getElementsByTagName("ymax")[0].firstChild.data
        bndbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
        bndboxs.append(bndbox)

    return bndboxs


class TestMetalTrain(unittest.TestCase):

    def test_localize_patches(self):
        '''
        Comprueba la etiqueta y que al menos un bounding-box intersecciones con el del ground-truth de una de las
        imágenes
        :return:
        '''
        # Cambiar path donde esten situadas las imagenes
        file_path = '../../NEU-DET/'
        # Se cargan todas las imagenes de tipo patches
        paths_images = glob.glob(file_path + 'IMAGES/patches*.jpg')
        
        total = 0
        intersectionTotal = 0

        for path_image in paths_images:
            file_name = os.path.basename(path_image)
            file_name = os.path.splitext(file_name)[0]
            # Cargamos el xml correspondiente de la imagen
            path_anno = file_path + 'ANNOTATIONS/' + file_name + '.xml'

            bndboxs = __get_gt_bndbox__(path_anno)
            patches_localizator = Patches_localizator()
            bb_results = patches_localizator.localize(path_image)
            # cv.waitKey(0)
            
            # Suma de todas las medias de IOU de cada imagen
            intersectionTotal += __get_intersection__(bb_results, bndboxs)
            total += 1
        print("Patches score", intersectionTotal / total)

        #Si la media de IOU es mas de un 50% el test es correcto
        self.assertTrue((intersectionTotal / total) > 0.50)

    def test_localize_scratches(self):
        '''
        Comprueba la etiqueta y que al menos un bounding-box intersecciones con el del ground-truth de una de las
        imágenes
        :return:
        '''
        # Cambiar path donde esten las imagenes de Scratches
        file_path = '../../NEU-DET/'
        # Se cargantodas las imagenes de Scratches
        paths_images = glob.glob(file_path + 'IMAGES/scratches*.jpg')
        total = 0
        intersectionTotal = 0

        for path_image in paths_images:
            file_name = os.path.basename(path_image)
            file_name = os.path.splitext(file_name)[0]
            # Cargamos el path del xml para la imagen correspondiente
            path_anno = file_path + 'ANNOTATIONS/' + file_name + '.xml'
            scratch_localizator = Scratch_localizator()
            bb_results = scratch_localizator.localize(path_image)
            # cv.waitKey(0)

            bndboxs = __get_gt_bndbox__(path_anno)
            intersectionTotal += __get_intersection__(bb_results, bndboxs)
            total += 1
        print("Scratches score", intersectionTotal / total)
        self.assertTrue((intersectionTotal / total) > 0.50)

    def test_classifier(self):
        '''
        Comprueba que la tasa de acierto de las etiquetas supere un umbral sobre unas imágenes de test
        :return:
        '''

        miu = MetalImperfectionsUtil()
        mi = MetalImperfections()
        test_files = miu.read_csv_file('./CNN_UTIL/mi_test.csv')
        dir_path='../../NEU-DET/IMAGES'

        # print(test_files)

        success = 0
        failure = 0
        total = 0
        for test_file in test_files:
            path_image = os.path.join(dir_path,test_file[0])
            label, bounding_boxes = mi.recognize(path_image)
            # print(path_image,test_file[0], label)
            total += 1
            if label == test_file[1]:
                success += 1
                print('success: ', success, ' de ', total)
            else:
                failure += 1
                print('failure: ', failure, ' de ', total)

        total = success + failure

        print('Score: ', str(1.0 * success / total))

        self.assertTrue(1.0 * success / total > 0.95)


if __name__ == '__main__':
    unittest.main()
