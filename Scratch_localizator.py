import cv2 as cv
import numpy as np
from Localizator import Localizator

class Scratch_localizator(Localizator):

    def localize(self, path_image):
        '''
        Method to localize scratches in a image
        :param path_image: Path to image
        :return: List with the bounding boxes
        '''
        # Read the image
        image = cv.imread(path_image)

        # Convert the image to gray scale
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        filteredImage = cv.blur(gray, (13, 13))
        filteredImage = cv.subtract(gray, filteredImage)
        # cv.imshow('sub', filteredImage)

        # Binarize the image 
        (ret, thresholdImage) = cv.threshold(filteredImage, 5, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow('trhes', thresholdImage)

        # Opening and closing morphological operations to erease noise 
        kernel = np.ones((2,2))
        opening = cv.morphologyEx(thresholdImage, cv.MORPH_OPEN, kernel)
        kernel = np.ones((3,3))
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        # Dilate shapes to connect broken lines 
        kernel = np.ones((1, 10), np.uint8)
        closing = cv.dilate(closing, kernel, iterations=1)
        kernel = np.ones((10, 1), np.uint8)
        closing = cv.dilate(closing, kernel, iterations=1)
        # cv.imshow('dilate', closing)
        
        # With the scratches in white we find the shapes contours
        contours, hierachy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            if w > 30 or h > 30:
                # Print the bounding box in the image to show it
                cv.rectangle(image,(x,y),(x+w,y+h),(200,0,0),2)

                # Append the bounding box to the return list 
                bounding_boxes.append((x, y, x+w, y+h))
        # cv.imshow('printed', image)
        
        # Return the bounding boxes list
        return bounding_boxes