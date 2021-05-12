import argparse
import os
from SafetyCheck import RecognizerMetalImperfections
import cv2

if __name__ == '__main__':
    # Read the images path argument
    parser = argparse.ArgumentParser(description='Path to images')
    parser.add_argument('images_path', metavar='images_path', type=str)
    args = parser.parse_args()

    #List the files in the path
    images = os.listdir(args.images_path)

    #Create the recognizer class instance
    metalImperfections = RecognizerMetalImperfections()

    for image in images:
        # Recognize the image getting the imperfection type and the bounding boxes
        image_read = cv2.imread(args.images_path + image, cv2.IMREAD_COLOR)
        label, bndboxs = metalImperfections.recognize(image_read)

        # Read the image to show it
        #image_read = cv2.imread(args.images_path + image)
        print("Imagen:", args.images_path + image, "Label y bndboxs", label, bndboxs)
        for x, y, xmax, ymax in bndboxs:
            # Print the bounding boxes in the image to show it
            cv2.rectangle(image_read,(x,y),(xmax,ymax),(200,0,0),2)
        cv2.imshow(label+'. Pulsar "q" para salir', image_read)
        # Wait for user key pressed 
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()
        
    cv2.destroyAllWindows()
        
