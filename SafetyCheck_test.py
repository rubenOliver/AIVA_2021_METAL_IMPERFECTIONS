# This code has been carried out for the Applications subject of the
# Master's Degree in Computer Vision at the Rey Juan Carlos University
# of Madrid.
# Contains the unit tests for the SafetyCheck library
# Date: April 2021
# Authors: Rubén Oliver, Ismael Linares and Juan Luis Carrillo

import unittest
from SafetyCheck import RecognizerMetalImperfections
import os
from xml.dom import minidom
from SafetyCheckUtil import MetalImperfectionsUtil
import glob
import numpy as np
from Patches_localizator import Patches_localizator
from Scratch_localizator import Scratch_localizator


def __find_index__(target, array):
    for i,row in enumerate(array):
        for j,value in enumerate(row):
            if value == target:
                return (i, j)

def __get_intersection__(bb_results, bndboxs):
    '''
    Calculate the intersections of two bounding-boxe sets.
    :param bb_results: Detector bounding-boxes
    :param bndboxs: Ground-truth bounding-boxes
    :return: Average IOU for all intersections
    '''

    # List with the IOU of each bndbox of the ground thruth with each prediction
    bb_intersections = []
    for bndbox in bndboxs:
        auxInter = []
        for bb_result in bb_results:
            xA = max(bb_result[0], bndbox[0])
            yA = max(bb_result[1], bndbox[1])
            xB = min(bb_result[2], bndbox[2])
            yB = min(bb_result[3], bndbox[3])

            # Area of ​​intersection between rectangles
            interArea = max(0, (xB - xA + 1)) * max(0,(yB - yA + 1))

            # Area for each bounding box for ground truth and prediction
            boxAArea = (bb_result[2] - bb_result[0] + 1) * (bb_result[3] - bb_result[1] + 1)
            boxBArea = (bndbox[2] - bndbox[0] + 1) * (bndbox[3] - bndbox[1] + 1)

            # Calculation of IOU for the degree of success
            iou = interArea / float(boxAArea + boxBArea - interArea)
            auxInter.append(iou)
        bb_intersections.append(auxInter)
    final_intersections = []

    gt_len = len(bndboxs)
    for i in range(0, gt_len):
        if not bb_intersections[0]:
            final_intersections.append(0.0)
        else:
            # Maximo IOU de todos los bndbox
            maxValue = np.max(bb_intersections)
            i, j = __find_index__(maxValue, bb_intersections)

            #
            # The GT bndbox and the column with the predicted bndbox are removed from the list
            # to only have a match for each GT bndbox
            del bb_intersections[i]
            if bb_intersections:
                for row in bb_intersections:
                        del row[j]
            final_intersections.append(maxValue)
    return sum(final_intersections) / len(final_intersections)

def __get_gt_bndbox__(path_anno):
    '''
    Get the bounding-boxes of the ground-truth
    :param path_anno: Annotations documents path
    :return: Ground-truth bounding-boxes list
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
        Check the label and that at least one bounding-box intersects
        with the one of the ground-truth of one of the images
        :return:
        '''

        # Path of image files
        file_path = '../../NEU-DET/'

        # Load all images with label patches
        paths_images = glob.glob(file_path + 'IMAGES/patches*.jpg')
        
        total = 0
        intersectionTotal = 0

        for path_image in paths_images:
            file_name = os.path.basename(path_image)
            file_name = os.path.splitext(file_name)[0]

            # Load .xml file with image bndboxs
            path_anno = file_path + 'ANNOTATIONS/' + file_name + '.xml'

            bndboxs = __get_gt_bndbox__(path_anno)
            patches_localizator = Patches_localizator()
            bb_results = patches_localizator.localize(path_image)
            
            # Sum of all IOU means for each image
            intersectionTotal += __get_intersection__(bb_results, bndboxs)
            total += 1
        print("Patches score", intersectionTotal / total)

        # If the mean IOU is more than 50% the test is correct
        self.assertTrue((intersectionTotal / total) > 0.50)

    def test_localize_scratches(self):
        '''
        Check the label and that at least one bounding-box intersects with the one of the
        ground-truth of one of the images
        :return:
        '''

        # Path of image files
        file_path = '../../NEU-DET/'

        # Load all Scratches images
        paths_images = glob.glob(file_path + 'IMAGES/scratches*.jpg')
        total = 0
        intersectionTotal = 0

        for path_image in paths_images:
            file_name = os.path.basename(path_image)
            file_name = os.path.splitext(file_name)[0]

            # Load .xml file with image bndboxs
            path_anno = file_path + 'ANNOTATIONS/' + file_name + '.xml'
            scratch_localizator = Scratch_localizator()
            bb_results = scratch_localizator.localize(path_image)

            bndboxs = __get_gt_bndbox__(path_anno)
            intersectionTotal += __get_intersection__(bb_results, bndboxs)
            total += 1
        print("Scratches score", intersectionTotal / total)
        self.assertTrue((intersectionTotal / total) > 0.50)

    def test_classifier(self):
        '''
        Check that the hit rate of the labels exceeds a threshold on some test images
        :return:
        '''

        miu = MetalImperfectionsUtil()
        mi = RecognizerMetalImperfections()
        test_files = miu.read_csv_file('./CNN_UTIL/mi_test.csv')
        dir_path='../../NEU-DET/IMAGES'

        success = 0
        failure = 0
        total = 0
        for test_file in test_files:
            path_image = os.path.join(dir_path,test_file[0])
            label, bounding_boxes = mi.recognize(path_image)
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
