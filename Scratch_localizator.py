import cv2 as cv
import numpy as np
from Localizator import Localizator

class Scratch_localizator(Localizator):

    def localize(self, path_image):
        image = cv.imread(path_image)
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        filteredImage = cv.blur(gray, (13, 13))
        filteredImage = cv.subtract(gray, filteredImage)
        # cv.imshow('sub', filteredImage)
        (ret, thresholdImage) = cv.threshold(filteredImage, 5, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow('trhes', thresholdImage)
        kernel = np.ones((2,2))
        opening = cv.morphologyEx(thresholdImage, cv.MORPH_OPEN, kernel)
        kernel = np.ones((3,3))
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        kernel = np.ones((1, 10), np.uint8)
        closing = cv.dilate(closing, kernel, iterations=1)
        kernel = np.ones((10, 1), np.uint8)
        closing = cv.dilate(closing, kernel, iterations=1)
        # cv.imshow('dilate', closing)
        contours, hierachy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            if w > 30 or h > 30:
                # roi=image[y:y+h,x:x+w]
                cv.rectangle(image,(x,y),(x+w,y+h),(200,0,0),2)
                bounding_boxes.append((x, y, x+w, y+h))
        cv.imshow('printed', image)
        return bounding_boxes