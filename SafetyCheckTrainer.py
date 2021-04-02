# Programa utilizado en la práctica de Reconocimiento de Patrones
# Autora: Vanesa Lomas
# Modificado por: Juan Luis Carrillo
#
# Comando:
# python train_autoencoder.py --filtros 32 64 --n_epochs 100 --tamano_batch 32 --carpeta_models ./models



# Cargar librerías y módulos
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Activation, \
    Flatten, Reshape, BatchNormalization, Conv2DTranspose, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
from SafetyCheckUtil import MetalImperfectionsUtil
from tensorflow.keras.models import load_model


class MetalImperfectionsTrainer:
    def __init__(self):
        self.num_classes = 4
        self.__gpu_setup()
        self.miu = MetalImperfectionsUtil()
        self.train_files = self.miu.read_csv_file('./mi_train.csv')
        self.test_files = self.miu.read_csv_file('./mi_test.csv')
        self.dir_path='./NEU-DET/IMAGES'

    def __gpu_setup(self):
        config=tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.compat.v1.Session(config=config)

    def shuffle_data(self, x_data, y_data):
        shuffled_indices=np.random.permutation(len(x_data))
        return x_data[shuffled_indices], y_data[shuffled_indices]


    def make_model(self, x_train):
        N_train, dim0, dim1=x_train.shape
        # - Input layer
        x=Input(shape=(dim0, dim1, 1))

        # - Convolution+Pooling layers
        h=Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        h=MaxPooling2D((2, 2))(h)
        h=Conv2D(16, (3, 3), activation='relu', padding='same')(h)
        h=MaxPooling2D((2, 2))(h)
        h=Conv2D(32, (3, 3), activation='relu', padding='same')(h)
        z=MaxPooling2D((2, 2))(h)

        # - Classification header
        z=Flatten()(z)
        z=Dense(64, activation='relu')(z)
        y=Dense(self.num_classes, activation='softmax')(z)

        # - Put all in a model and compile
        cnn=Model(x, y)
        cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        cnn.summary()
        return cnn

    def augment_data(self, x_train, y_train):
        x_rot90 = np.rot90(x_train, axes=(1, 2))
        x_rot180 = np.rot90(x_rot90, axes=(1, 2))
        x_rot270 = np.rot90(x_rot180, axes=(1, 2))
        # x_flip_v = np.flip(x_train, axis=1)
        # x_flip_h = np.flip(x_train, axis=2)
        # x_flip_12 = np.flip(x_train, axis=(1,2))
        # x_train_new = np.vstack((x_train,x_rot90,x_rot180,x_rot270,x_flip_v,x_flip_h,x_flip_12))
        # y_train_new = np.hstack((y_train,y_train,y_train,y_train,y_train,y_train,y_train))

        x_train_new = np.vstack((x_train,x_rot90,x_rot180,x_rot270))
        y_train_new = np.hstack((y_train,y_train,y_train,y_train))
        return x_train_new, y_train_new


    def train_model(self, carpeta_models, n_epochs, batch_size):
        # Se cargan los datos de entrenamiento
        x_train, y_train = self.miu.read_data(self.dir_path, self.train_files)

        # Se normalilzan los datos
        x_train = x_train.astype('float32') / 255.0


        # Se mezclan los datos
        x_train, y_train = self.shuffle_data(x_train, y_train)
        # print(x_train.shape)

        # Se usa un 20% de los datos de entrenamiento para validación
        x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              random_state=42)

        # print(x_train.shape, y_train.shape)
        # print(y_train[0])
        # cv2.imshow('antes',x_train[0,:,:])
        x_train, y_train = self.augment_data(x_train, y_train)
        # print(y_train[1152])
        # cv2.imshow('despues', x_train[x_train.shape[0]-1152, :, :])
        # print(x_train.shape, y_train.shape)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        # Se crea el modelo de red neuronal
        cnn = self.make_model(x_train)

        # Se reorganizan los datos de entrenamiento
        n_train, dim0, dim1 = x_train.shape
        x_train = x_train.reshape((n_train, dim0, dim1, 1))
        y_train = to_categorical(y_train, num_classes=self.num_classes)

        # Se reorganizan los datos de validación
        n_valid, dim0, dim1 = x_valid.shape
        x_valid = x_valid.reshape((n_valid, dim0, dim1, 1))
        y_valid = to_categorical(y_valid, num_classes=self.num_classes)


        tensorboard = TensorBoard(log_dir="logs/",
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)


        checkpoint_filepath=os.path.join(carpeta_models,
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

        # Se guardan las gráficas de loss y accuracy
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


    def predict(self, path_model):
        x_test, y_test = self.miu.read_data(self.dir_path, self.test_files)

        # Normalizar los valores
        x_test = x_test.astype('float32') / 255.0

        n_test, dim0, dim1 = x_test.shape
        x_test = x_test.reshape((n_test, dim0, dim1, 1))
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)


        # print(path_model)
        cnn = load_model(path_model)

        # Evaluación del modelo con el método evaluate
        resultados = cnn.evaluate(x_test, y_test_cat)
        print('-----RESULTADOS-----')
        print('Test loss:', resultados[0])
        print('Test accuracy:', resultados[1])

        # Se aplica el método predict sobre los ejemplos de test
        yhat = cnn.predict(x_test)
        k=300
        print(y_test[k], '<-- y')
        print(np.argmax(yhat[k, :]), '<-- yhat')

        y_pred=np.argmax(yhat[:, :], axis=1)

        count=yhat.shape[0]
        conf_mat=confusion_matrix(y_test, y_pred)
        print(conf_mat)
        hits=conf_mat[0, 0]+conf_mat[1, 1]+conf_mat[2, 2]+conf_mat[3, 3]
        fails=count-hits
        # hits_list.append(hits)
        strlog="Fold %d: HITS = %d, FAILS = %d"%(count, hits, fails)
        print(strlog)

        # cv2.imshow('imagen', x_test[k, :, :])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    directorio_model = './models'
    try:
        os.stat(directorio_model)
    except:
        os.mkdir(directorio_model)

    mif = MetalImperfectionsTrainer()

    mif.train_model('./models', 100, 32)

    # mif.predict('./models/weights_improvement.52-0.0150.h5')