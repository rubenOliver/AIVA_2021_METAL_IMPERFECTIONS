import cv2 as cv
import numpy as np
from Localizator import Localizator
class Patches_localizator(Localizator):
    
    def localize(self, path_image):
        image = cv.imread(path_image)

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # cv.imshow('hsv', hsv)
        binary_img_hsv = cv.adaptiveThreshold(hsv[:,:, 2], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 71, 15)
        # cv.imshow('gaussian_hsv', binary_img_hsv)
        kernel = np.ones((5,5))
        opening_mean = cv.morphologyEx(binary_img_hsv, cv.MORPH_OPEN, kernel)
        # cv.imshow('open', opening_mean)
        kernel = np.ones((3,3))
        opening_mean = cv.morphologyEx(opening_mean, cv.MORPH_CLOSE, kernel)

        # kernel = np.ones((3,3),np.uint8)
        # opening_mean = cv.dilate(opening_mean,kernel,iterations = 1)

        contours, hierachy = cv.findContours(opening_mean, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            if w > 25 or h > 25:
                # roi=image_gauss[y:y+h,x:x+w]
                cv.rectangle(image,(x,y),(x+w,y+h),(200,0,0),2)
                bounding_boxes.append((x, y, x+w, y+h))
        # cv.imshow('printed', image)
        # cv.imshow('closing gauss', opening_mean)
        return bounding_boxes
