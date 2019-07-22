import os
import cv2
import sys
import pandas
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization, concatenate
from sklearn.model_selection import train_test_split

def quit_with(err):
    print(err)
    sys.exit()

def load_data(imgs=True):
    frame = pandas.read_csv("data/data.csv", header=None)
    data = frame.values
    X_scalar = data[:, 0:6]
    Y = data[:, 6:] / (1920, 1080)

    if imgs:
        X_scalar /= np.amax(X_scalar, axis=0)
        img_folder = 'data/imgs'
        num_pairs = int(len([name for name in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, name))]) / 2)
        X_cnn_left = [0]*num_pairs
        X_cnn_right = [0]*num_pairs
        for fname in os.listdir(img_folder):
            idx = int("".join(filter(str.isdigit, fname)))
            img = cv2.imread(os.path.join(img_folder, fname), cv2.IMREAD_GRAYSCALE)  / 255.0
            if img is None:
                quit_with("Error loading image data.")
            else:
                if 'l' in fname:
                    X_cnn_left[idx] = img
                elif 'r' in fname:
                    X_cnn_right[idx] = img
                else:
                    quit_with("Non compliant image: " + fname)
        return np.array(X_scalar), np.array(X_cnn_left), np.array(X_cnn_right), np.array(Y)
    else:
        return np.array(X_scalar), np.array(Y)

def build_convnet(input_conv):
    convnet = Conv2D(32, use_bias=False, kernel_size=3, input_shape=(32, 32, 1))(input_conv)
    convnet = BatchNormalization()(convnet)
    convnet = Activation("relu")(convnet)
    convnet = Conv2D(32, kernel_size=3, activation='relu')(convnet)
    convnet = MaxPooling2D(pool_size=(2, 2))(convnet)
    convnet = Conv2D(64, kernel_size=3, activation='relu')(convnet)
    convnet = Flatten()(convnet)
    convnet = Dense(512, activation='relu')(convnet)
    convnet = Dropout(0.5)(convnet)
    convnet = Dense(128, activation='relu')(convnet) # balance with densenet
    convnet = Dense(64, activation='relu')(convnet) # balance with densenet
    convnet = Dropout(0.5)(convnet)
    return Model(inputs=input_conv, outputs=convnet)

def build_densenet(input_dense):
    densenet = Dense(64, activation='relu')(input_dense)
    densenet = Dense(32, activation='relu')(densenet)
    densenet = Dropout(0.5)(densenet)
    densenet = Dense(32, activation='relu')(densenet)
    return Model(inputs=input_dense, outputs=densenet)

def build_model():
    input_conv_left = Input(shape=(32, 32, 1))
    input_conv_right = Input(shape=(32, 32, 1))
    input_dense = Input(shape=(6,))
    convnet_left = build_convnet(input_conv_left)
    convnet_right = build_convnet(input_conv_right)
    densenet = build_densenet(input_dense)
    model = concatenate([convnet_left.output, convnet_right.output, densenet.output])
    model = Dense(16, activation='relu')(model)
    model = Dense(4, activation='relu')(model)
    model = Dense(2, activation='linear')(model)
    model = Model(inputs=[convnet_left.input, convnet_right.input, densenet.input], outputs=model)
    return model

def main(preload_model=False):
    X_scalar, X_cnn_left, X_cnn_right, Y = load_data()
    X_scalar_train, X_scalar_test, X_cnn_left_train, X_cnn_left_test, X_cnn_right_train, X_cnn_right_test, Y_train, Y_test = train_test_split(X_scalar, X_cnn_left, X_cnn_right, Y, test_size=0.25)
    X_cnn_left_train = np.expand_dims(X_cnn_left_train, axis=3)
    X_cnn_right_train = np.expand_dims(X_cnn_right_train, axis=3)
    X_cnn_left_test = np.expand_dims(X_cnn_left_test, axis=3)
    X_cnn_right_test = np.expand_dims(X_cnn_right_test, axis=3)
    model = load_model('data/model.hdf5') if preload_model else build_model()
    opt = Adam(lr=1e-4, decay=1e-4 / 200)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', 'acc'])
    print("Successfully compiled model, begin training.")
    model.fit([X_cnn_left_train, X_cnn_right_train, X_scalar_train], Y_train,
                #validation_data=([X_cnn_left_test, X_cnn_right_test, X_scalar_test], Y_test),
                epochs=50, batch_size=64)
    print("Finished training, evaluating.")
    results = model.evaluate([X_cnn_left_test, X_cnn_right_test, X_scalar_test], Y_test, batch_size=64)
    print("Test loss, acc: ", results)
    print("Dumping weights to disk.")
    model.save('data/model.hdf5')

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main(input("Load weights [y/n] ") == 'y')