# This code has been carried out for the Applications subject of the
# Master's Degree in Computer Vision at the Rey Juan Carlos University
# of Madrid.
# Date: April 2021
# Authors: Rubén Oliver, Ismael Linares and Juan Luis Carrillo

import numpy as np
import csv
import os
import cv2

class MetalImperfectionsUtil:
    '''
    This class encapsulates a series of functionality that will be useful for the rest of the SafetyCheck library
    '''

    valid_labels = ['inclusion', 'patches', 'scratches'] # Classes to detect
    other_label = 'other' # Nlternative class name

    def read_one_image(self, file_path):
        '''
        Read an image in correct format to work with neural network
        :param file_path: Path of the image
        :return: Return a numpy array with the image in grayscale format
        '''
        im_gray=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        x_data=np.zeros((1, im_gray.shape[0], im_gray.shape[1]), np.uint8)
        x_data[0,:,:] = im_gray
        return x_data

    def read_data(self, dir_path, data_files):
        '''
        Read data images from the a folder
        :param dir_path: Path of the images folder
        :param data_files: List of tuples with file name and extension of image, and label
        :return: Feature and class data
        '''

        # Read first image to get image dimension
        file_path=os.path.join(dir_path,data_files[0][0])
        im_gray=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Create a numpy array with size of examples and image dimension
        x_data = np.zeros((len(data_files),im_gray.shape[0], im_gray.shape[1]),np.uint8)

        # Create a numpy array with size of examples
        y_data = np.zeros((len(data_files),),np.uint8)

        # Load image data
        for i in range(len(data_files)):
            file_path = os.path.join(dir_path, data_files[i][0])
            im_gray=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            x_data[i,:,:] = im_gray

            # Store label number from text label
            y_data[i] = self.get_label_number(data_files[i][1])

        return x_data, y_data



    def count_by_label(self, y_data, label):
        '''
        Count the number of examples of the label
        :param y_data: Numpy array with label numbers of data
        :param label: Label text
        :return: Number of examples of the label
        '''
        label_number = self.get_label_number(label)
        return np.count_nonzero(y_data == label_number)


    def get_valid_label(self, label):
        '''
        Check if the label is within the correct ones
        :param label: The text label
        :return: The text label, if this is a valid label, or 'other' in other case
        '''
        for valid_label in self.valid_labels:
            if label == valid_label:
                return label
        return self.other_label


    def get_label_number(self, label):
        '''
        Get the number label from text label
        :param label: The text label
        :return: The number associated with the label
        '''
        counter = 0
        for valid_label in self.valid_labels:
            counter += 1
            if label == valid_label:
                return counter
        return 0


    def get_label_text(self, label_index):
        '''
        Get the text label from number label
        :param label_index: Number associated with the label
        :return: Text label, if it is a correct label, or 'other' in other case
        '''
        index = label_index - 1
        if index>=0 and index<len(self.valid_labels):
            return self.valid_labels[index]
        return self.other_label


    def read_csv_file(self, path_cvs):
        '''
        Read a .cvs file with the image names and label
        :param path_cvs: Path of the .cvs file
        :return: A tuple width names of images and label of each image
        '''
        data_files = []
        with open(path_cvs, 'r') as file:
            reader=csv.reader(file, delimiter=',')
            for row in reader:
                label = row[0].strip()
                for i in range(1, len(row)):
                    data_files.append((label+'_'+row[i].strip()+'.jpg', self.get_valid_label(label)))
        return data_files


    def split_data_and_write_file(self, path_cvs_test, path_cvs_train, type_labels, test_ratio):
        '''
        Split a set of hypothetical images of different classes into a training set and a test set.
        Store these sets in two diferent .csv files
        :param path_cvs_test: Path of the .csv file for test images
        :param path_cvs_train: Path of the .csv file for train images
        :param type_labels: List of all labels (valid and not valid labels)
        :param test_ratio: Ratio of test images over total images
        :return:
        '''
        f_test=open(path_cvs_test, "w")
        f_train=open(path_cvs_train, "w")
        for type_label in type_labels:
            label, examples_number = type_label
            test_number = int(examples_number*test_ratio)
            shuffled_indices=np.random.permutation(examples_number)
            test_indexes=shuffled_indices[:test_number]
            train_indexes=shuffled_indices[test_number:]
            f_test.write(label)
            for test_index in test_indexes:
                f_test.write(', '+str(test_index+1))
            f_test.write('\n')

            f_train.write(label)
            for train_index in train_indexes:
                f_train.write(', '+str(train_index+1))
            f_train.write('\n')

        f_test.close()
        f_train.close()



if __name__=='__main__':
    # Split data to train and test
    miu = MetalImperfectionsUtil()
    number_images = 300
    type_labels=[('crazing', number_images),
                 ('inclusion', number_images),
                 ('patches', number_images),
                 ('pitted_surface', number_images),
                 ('rolled-in_scale', number_images),
                 ('scratches', number_images)]
    miu.split_data_and_write_file('./mi_test.csv', './mi_train.csv', type_labels, 0.2)


