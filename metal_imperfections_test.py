"""
Fichero con los test unitarios del entrenamiento y del reconocedor
"""

import unittest
import tensorflow as tf
from metal_imperfections import MetalImperfections

class TestMetalTrain(unittest.TestCase):
    """
    Test unitarios de los reconocedores de defectos en metales
    """
    def test_splitData(self):
        metal_imperfections = MetalImperfections()
        metal_imperfections.splitData('./')

        # A continuación se comprueba que se han creado los distintos ficheros csv
        # y que el número de imágenes en cada fichero es correcto
        pass

    def test_loadTestData(self):
        metal_imperfections = MetalImperfections()
        x_test, y_test = metal_imperfections.loadTestData('./')

        # A continuación, se comprueba que el número de imágenes test es correcto
        pass

    def test_loadTrainData(self):
        metal_imperfections = MetalImperfections()
        x_train, y_train = metal_imperfections.loadTrainData('./')

        # A continuación, se comprueba que el número de imágenes train es correcto
        pass

    def test_loadValidData(self):
        metal_imperfections = MetalImperfections()
        x_valid, y_valid = metal_imperfections.loadValidData('./')

        # A continuación, se comprueba que el número de imágenes valid es correcto
        pass

    def test_classifier(self):
        metal_imperfections = MetalImperfections()
        x_test, y_test = metal_imperfections.loadTestData('pathTestImages.csv')
        x_test = x_test / 255.0
        model = tf.keras.models.model_from_json('metal_imperfections.json')
        model.local_weights('metal_imperfections.h5')
        model.compile(optimizer='adams', metrics=['accuracy'])
        score = model.evaluate(x_test, y_test)
        self.assertTrue(score[1] > 0.95)


if __name__ == '__main__':
    unittest.main()
