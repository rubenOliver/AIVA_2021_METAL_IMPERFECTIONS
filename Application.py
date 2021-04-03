import argparse
import os
from SafetyCheck import MetalImperfections
import cv2 as cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path to images')
    parser.add_argument('images_path', metavar='images_path', type=str)
    args = parser.parse_args()
    images = os.listdir(args.images_path)
    metalImperfections = MetalImperfections()

    for image in images:
        label, bndboxs = metalImperfections.recognize(args.images_path + image)
        image_read = cv.imread(args.images_path + image)
        print("Imagen:", args.images_path + image, "Label y bndboxs", label, bndboxs)
        for x, y, xmax, ymax in bndboxs:
            cv.rectangle(image_read,(x,y),(xmax,ymax),(200,0,0),2)
        cv.imshow(label, image_read)
        cv.waitKey(0)
        cv.destroyAllWindows()
        