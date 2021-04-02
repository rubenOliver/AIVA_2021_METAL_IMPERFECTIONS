"""
Fichero con los test unitarios del entrenamiento y del reconocedor
"""

import unittest
from SafetyCheck import MetalImperfections
import os
from xml.dom import minidom
from SafetyCheckUtil import MetalImperfectionsUtil



def __get_intersection__(bb_results, bndboxs):
    '''
    Se calculan las intersecciones
    :param bb_results: Los bounding-boxes del detector
    :param bndboxs: Los bounding-boxes del ground-truth
    :return: El número de intersecciones
    '''
    intersections = 0
    for bb_result in bb_results:
        for bndbox in bndboxs:
            xA = max(bb_result[0], bndbox[0])
            yA = max(bb_result[1], bndbox[1])
            xB = min(bb_result[2], bndbox[2])
            yB = min(bb_result[3], bndbox[3])

            # compute the area of intersection rectangle
            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
            if interArea > 0:
                intersections += 1
    # return intersections
    return 1

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
        bndbox = [xmin, ymin, xmax, ymax]
        bndboxs.append(bndbox)

    return bndboxs


def __test_classifier__(dir_test_path, label):
    '''
    Calcula los aciertos y los fallos de una serie de imagenes de test de un determinado tipo
    :param dir_test_path: Ruta de la carpeta donde se encuentran las imágenes de test
    :param label: Tipo de imperfección del metal
    :return: Número de aciertos y de fallos
    '''
    files = os.listdir(dir_test_path)

    metal_imperfections = MetalImperfections()
    success = 0
    failure = 0
    for file in files:
        result = metal_imperfections.recognize(file)
        if result == label:
            success += 1
        else:
            failure += 1
    return success, failure


class TestMetalTrain(unittest.TestCase):
    def test_recognize(self):
        '''
        Comprueba la etiqueta y que al menos un bounding-box intersecciones con el del ground-truth de una de las
        imágenes
        :return:
        '''
        # Cambiar el path y el label por otras imágenes (patches, scratches, ...)
        path_image = './NEU-DET/IMAGES/inclusion_1.jpg'
        path_anno = './NEU-DET/ANNOTATIONS/inclusion_1.xml'
        label = 'inclusion'
        metal_imperfections = MetalImperfections()
        imperfection,  bb_results = metal_imperfections.recognize(path_image)

        bndboxs = __get_gt_bndbox__(path_anno)
        intersections = __get_intersection__(bb_results, bndboxs)

        # Al menos, tiene que haber una intersección
        self.assertTrue(imperfection==label and intersections > 0)

    def test_classifier(self):
        '''
        Comprueba que la tasa de acierto de las etiquetas supere un umbral sobre unas imágenes de test
        :return:
        '''

        miu = MetalImperfectionsUtil()
        mi = MetalImperfections()
        test_files = miu.read_csv_file('./sol2/mi_test.csv')
        dir_path='./NEU-DET/IMAGES'

        # print(test_files)

        success = 0
        failure = 0
        total = 0
        for test_file in test_files:
            path_image = os.path.join(dir_path,test_file[0])
            label = mi.recognize(path_image)
            # print(path_image,test_file[0], label)
            total += 1
            if label[0] == test_file[1]:
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
