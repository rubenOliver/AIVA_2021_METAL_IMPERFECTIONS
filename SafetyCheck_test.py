"""
Fichero con los test unitarios del entrenamiento y del reconocedor
"""

import unittest
# import tensorflow as tf
from SafetyCheck import MetalImperfections
import csv
import os
from xml.dom import minidom

class TestMetalTrain(unittest.TestCase):
    def test_getBndbox(self):
        path_image = './NEU-DET/IMAGES/inclusion_1.jpg'
        path_anno = './NEU-DET/ANNOTATIONS/inclusion_1.xml'
        label = 'inclusion'

        bndboxs = []
        doc = minidom.parse(path_anno)
        objects = doc.getElementsByTagName("object")
        for object in objects:
            xmin = object.getElementsByTagName("xmin")[0].firstChild.data
            ymin = object.getElementsByTagName("ymin")[0].firstChild.data
            xmax = object.getElementsByTagName("xmax")[0].firstChild.data
            ymax = object.getElementsByTagName("ymax")[0].firstChild.data
            bndbox = [xmin, ymin, xmax, ymax]
            # print(bndbox)
            bndboxs.append(bndbox)

        metal_imperfections = MetalImperfections()
        bb_results = metal_imperfections.getBndbox(path_image, label)


        # Se calculan las intersecciones
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

        # Al menos, tiene que haber una intersección
        self.assertTrue(intersections > 0)


    def test_recognize(self):
        # Cambiar el path y el label por otras imágenes (patches, scratches, ...)
        path_image = './NEU-DET/IMAGES/inclusion_1.jpg'
        label = 'inclusion'
        metal_imperfections = MetalImperfections()
        result = metal_imperfections.recognize(path_image)
        self.assertEqual(result, label)

    def test_classifier(self):
        dir_test_path = './NEU-DET/IMAGES/test/inclusion/'
        label = 'inclusion'
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
        total = success + failure
        if total == 0:
            self.assertFalse(total == 0)
        else:
            self.assertTrue(1.0 * success / total > 0.95)


if __name__ == '__main__':
    unittest.main()
