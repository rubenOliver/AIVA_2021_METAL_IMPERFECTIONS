import cv2 as cv
import numpy as np
from Localizator import Localizator
class Patches_localizator(Localizator):
    
    def localize(self, path_image):
        '''
        Method to localize patches in a image
        :param path_image: Path to image
        :return: List with the bounding boxes
        '''
        # Read the image
        # image = cv.imread(path_image)
        image = path_image.copy()
        # Convert image to hsv colour space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # cv.imshow('hsv', hsv)

        # Binarize image with adaptative threshold mean to highlight the darkest areas 
        binary_img_hsv = cv.adaptiveThreshold(hsv[:,:, 2], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 71, 15)
        # cv.imshow('gaussian_hsv', binary_img_hsv)

        # Opening and closing filter to erease noise
        kernel = np.ones((5,5))
        opening_mean = cv.morphologyEx(binary_img_hsv, cv.MORPH_OPEN, kernel)
        # cv.imshow('open', opening_mean)
        kernel = np.ones((3,3))
        opening_mean = cv.morphologyEx(opening_mean, cv.MORPH_CLOSE, kernel)

        # Find contours of the white shapes
        contours, hierachy = cv.findContours(opening_mean, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            if w > 25 or h > 25:
                # Print the bounding box in the image to show it
                cv.rectangle(image,(x,y),(x+w,y+h),(200,0,0),2)

                # Append the bounding box to the return list 
                bounding_boxes.append((x, y, x+w, y+h))
        # cv.imshow('printed', image)

        # Return the bounding boxes list
        return bounding_boxes
