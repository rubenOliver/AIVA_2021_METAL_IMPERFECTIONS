# This code has been carried out for the Applications subject of the
# Master's Degree in Computer Vision at the Rey Juan Carlos University
# of Madrid.
# Date: April 2021
# Authors: Rubén Oliver, Ismael Linares and Juan Luis Carrillo


from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from SafetyCheckUtil import MetalImperfectionsUtil
from tensorflow.keras.models import load_model


class MetalImperfectionsTrainer:
    '''
    This class is used for the training of the neural network that the recognizer will use
    '''

    def __init__(self):
        self.num_classes = 4 # Currently, four different classes are recognized:
                             # 'inclusion', 'patches', 'scratches' and 'other'
        self.__gpu_setup()
        self.miu = MetalImperfectionsUtil()
        self.train_files = self.miu.read_csv_file('./mi_train.csv') # .csv file with the identification numbers
                                                                    # of the training images
        self.test_files = self.miu.read_csv_file('./mi_test.csv') # .csv file with the identification numbers
                                                                  # of the test images
        self.dir_path='./NEU-DET/IMAGES' # Directory path where the images to be
                                         # used for training are located

    def __gpu_setup(self):
        '''
        This method is necessary to properly configure tensorflow.
        Without this call, this library may cause some failure
        :return:
        '''
        config=tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        tf.compat.v1.Session(config=config)

    def shuffle_data(self, x_data, y_data):
        '''
        This method shuffles the data to avoid that an entire block that feeds
        the neural network is of the same class
        :param x_data: Feature data
        :param y_data: Class data
        :return: Shuffled data
        '''
        shuffled_indices=np.random.permutation(len(x_data))
        return x_data[shuffled_indices], y_data[shuffled_indices]


    def make_model(self, x_train):
        '''
        Create the network model
        :param x_train: Feature data. It is used to get the dimension of the neural network
        :return: Neural network model.
        '''

        # Dimension feature data dimension
        N_train, dim0, dim1=x_train.shape

        # Input layer
        x=Input(shape=(dim0, dim1, 1))

        # Convolution+Pooling layers
        h=Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        h=MaxPooling2D((2, 2))(h)
        h=Conv2D(16, (3, 3), activation='relu', padding='same')(h)
        h=MaxPooling2D((2, 2))(h)
        h=Conv2D(32, (3, 3), activation='relu', padding='same')(h)
        z=MaxPooling2D((2, 2))(h)

        # Classification header
        z=Flatten()(z)
        z=Dense(64, activation='relu')(z)
        y=Dense(self.num_classes, activation='softmax')(z)

        # Put all in a model and compile
        cnn=Model(x, y)
        cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        cnn.summary()
        return cnn

    def augment_data(self, x_train, y_train):
        '''
        To improve network performance, data augmentation is carried out by rotating
        the original images.
        :param x_train: Feature data for training
        :param y_train: Class data for training
        :return: Augmentated data
        '''
        x_rot90 = np.rot90(x_train, axes=(1, 2))
        x_rot180 = np.rot90(x_rot90, axes=(1, 2))
        x_rot270 = np.rot90(x_rot180, axes=(1, 2))

        x_train_new = np.vstack((x_train,x_rot90,x_rot180,x_rot270))
        y_train_new = np.hstack((y_train,y_train,y_train,y_train))
        return x_train_new, y_train_new


    def train_model(self, model_folder, n_epochs, batch_size):
        '''
        This method performs the training. Internally,
        It creates the neural network model
        :param model_folder: Folder where the models .h5 will be saved
        :param n_epochs: Number of epochs for training
        :param batch_size: Size of batch data for training
        :return:
        '''
        # Load of training data
        x_train, y_train = self.miu.read_data(self.dir_path, self.train_files)

        # Data normalization
        x_train = x_train.astype('float32') / 255.0

        # Shuffling data
        x_train, y_train = self.shuffle_data(x_train, y_train)

        # 20% of the training data is used for validation
        x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=42)

        x_train, y_train = self.augment_data(x_train, y_train)

        # The neural network model is created
        cnn = self.make_model(x_train)

        # Training data is conveniently rearranged
        n_train, dim0, dim1 = x_train.shape
        x_train = x_train.reshape((n_train, dim0, dim1, 1))
        y_train = to_categorical(y_train, num_classes=self.num_classes)

        # Validation data is conveniently rearranged
        n_valid, dim0, dim1 = x_valid.shape
        x_valid = x_valid.reshape((n_valid, dim0, dim1, 1))
        y_valid = to_categorical(y_valid, num_classes=self.num_classes)

        # Preparing the callbacks functions
        tensorboard = TensorBoard(log_dir="logs/",
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)


        checkpoint_filepath=os.path.join(model_folder,
                                         'weights_improvement.{epoch:02d}-{val_loss:.4f}.h5')

        model_checkpoint = ModelCheckpoint(checkpoint_filepath,
                                           monitor='val_loss',
                                           save_best_only=True,
                                           mode='min',
                                           save_freq='epoch')

        list_callbacks = [tensorboard, model_checkpoint]

        # Se entrena la red neuronal
        history = cnn.fit(x_train,
                          y_train,
                          validation_data=(x_valid, y_valid),
                          epochs=n_epochs,
                          batch_size=batch_size,
                          callbacks=list_callbacks)

        # Loss and accuracy graphs are saved
        plt.figure(1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig('Model_loss.png')

        plt.figure(2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig('Model_accuracy.png')


    def _predict(self, path_model):
        '''
        This method is used only for debugging purposes during the programming phase
        :param path_model: Path of the neural network .h5 model
        :return:
        '''

        # Load of test data
        x_test, y_test = self.miu.read_data(self.dir_path, self.test_files)

        # Data normalization
        x_test = x_test.astype('float32') / 255.0

        # Test data is conveniently rearranged
        n_test, dim0, dim1 = x_test.shape
        x_test = x_test.reshape((n_test, dim0, dim1, 1))
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)

        # Load de neural network .h5 model
        cnn = load_model(path_model)

        # Evaluación del modelo con el método evaluate
        resultados = cnn.evaluate(x_test, y_test_cat)
        print('-----RESULTS-----')
        print('Test loss:', resultados[0])
        print('Test accuracy:', resultados[1])

        # The predict method is applied to the test examples
        yhat = cnn.predict(x_test)
        k=300 # An example
        print(y_test[k], '<-- y')
        print(np.argmax(yhat[k, :]), '<-- yhat')

        # Calculating the confusion matrix
        y_pred=np.argmax(yhat[:, :], axis=1)
        count=yhat.shape[0]
        conf_mat=confusion_matrix(y_test, y_pred)
        print(conf_mat)
        hits=conf_mat[0, 0]+conf_mat[1, 1]+conf_mat[2, 2]+conf_mat[3, 3]
        fails=count-hits
        strlog="Fold %d: HITS = %d, FAILS = %d"%(count, hits, fails)
        print(strlog)


if __name__ == '__main__':
    directorio_model = './models' # Path of model folder
    try:
        # Checking that the folder exists. If the folder does not exist an exception occurs
        os.stat(directorio_model)
    except:
        # Making the folder
        os.mkdir(directorio_model)

    # The object is created
    mif = MetalImperfectionsTrainer()

    # Training neural network
    mif.train_model('./models', 100, 32)