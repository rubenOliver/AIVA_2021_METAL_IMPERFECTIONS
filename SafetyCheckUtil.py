import numpy as np
import csv
import os
import cv2

class MetalImperfectionsUtil:
    valid_labels = ['inclusion', 'patches', 'scratches']
    other_label = 'other'

    def read_one_image(self, file_path):
        im_gray=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        x_data=np.zeros((1, im_gray.shape[0], im_gray.shape[1]), np.uint8)
        x_data[0,:,:] = im_gray
        return x_data

    def read_data(self, dir_path, data_files):
        file_path=os.path.join(dir_path,data_files[0][0])

        im_gray=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        x_data = np.zeros((len(data_files),im_gray.shape[0], im_gray.shape[1]),np.uint8)
        y_data = np.zeros((len(data_files),),np.uint8)
        for i in range(len(data_files)):
            file_path = os.path.join(dir_path, data_files[i][0])
            im_gray=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            x_data[i,:,:] = im_gray
            y_data[i] = self.get_label_number(data_files[i][1])

        return x_data, y_data



    def count_by_label(self, y_data, label):
        label_number = self.get_label_number(label)
        return np.count_nonzero(y_data == label_number)


    def get_valid_label(self, label):
        for valid_label in self.valid_labels:
            if label == valid_label:
                return label

        return self.other_label

    def get_label_number(self, label):
        counter = 0
        for valid_label in self.valid_labels:
            counter += 1
            if label == valid_label:
                return counter

        return 0

    def get_label_text(self, label_index):
        index = label_index - 1
        if index>=0 and index<len(self.valid_labels):
            return self.valid_labels[index]

        return self.other_label

    def read_csv_file(self, path_cvs):
        data_files = []
        with open(path_cvs, 'r') as file:
            reader=csv.reader(file, delimiter=',')
            for row in reader:
                label = row[0].strip()
                for i in range(1, len(row)):
                    data_files.append((label+'_'+row[i].strip()+'.jpg', self.get_valid_label(label)))
        return data_files



    def split_data_and_write_file(self, path_cvs_test, path_cvs_train, type_labels, test_ratio):
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
    miu = MetalImperfectionsUtil()


    number_images = 300
    type_labels=[('crazing', number_images),
                 ('inclusion', number_images),
                 ('patches', number_images),
                 ('pitted_surface', number_images),
                 ('rolled-in_scale', number_images),
                 ('scratches', number_images)]
    miu.split_data_and_write_file('./mi_test.csv', './mi_train.csv', type_labels, 0.2)







    # print('Los ficheros para test son:')
    test_files = miu.read_csv_file('./mi_test.csv')
    print(test_files)




    #
    # print('\Los ficheros para entrenamiento son: ')
    # train_files = miu.read_csv_file('./mi_train.csv')
    # print(train_files)
    #
    # print('Número de etiquetas de other: ',miu.count_by_label(train_files, 'other'))
    # # print(train_files[0][0])
    #
    dir_path = './NEU-DET/IMAGES'


    example = 4
    x_data, y_data = miu.read_data(dir_path, test_files)
    print(y_data)

    print('Número de etiquetas de other: ', miu.count_by_label(y_data, 'inclusion'))
    # print(miu.get_label_text(y_data[example]))
    #
    #
    # # im_gray=cv2.imread(dir_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('image', x_data[example,:,:])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

