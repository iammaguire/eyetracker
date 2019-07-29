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

def load_data(imgs=True):
    frame = pandas.read_csv("data/data.csv", header=None)
    data = frame.values
    to_load = 20000
    X_scalar = data[:to_load, 0:6]
    Y = data[:to_load, 6:] / (1920, 1080)

    if imgs:
        X_scalar /= np.amax(X_scalar, axis=0)
        img_folder = 'data/imgs/'
        num_pairs = int(len([name for name in os.listdir(img_folder + 'l/') if os.path.isfile(os.path.join(img_folder + 'l/', name))]))
        X_cnn_left = [0]*to_load
        X_cnn_right = [0]*to_load
        loaded = 0
        for fname in os.listdir(img_folder + 'l/'):
            loaded += 1
            if loaded == to_load: break
            idx = loaded-1#int("".join(filter(str.isdigit, fname)))
            l_img = cv2.imread(os.path.join(img_folder + 'l/', fname), cv2.IMREAD_GRAYSCALE) / 255.0
            r_img = cv2.imread(os.path.join(img_folder + 'r/', fname), cv2.IMREAD_GRAYSCALE) / 255.0
            if loaded % 1000 == 0: print(loaded)
            if l_img is None or r_img is None:
                print("Error loading image data.")
                sys.exit()
            else:
                X_cnn_left[idx] = l_img
                X_cnn_right[idx] = r_img
        return np.array(X_scalar), np.array(X_cnn_left), np.array(X_cnn_right), np.array(Y)
    else:
        return np.array(X_scalar), np.array(Y)

def build_convnet(input_conv):
    convnet = Conv2D(8, kernel_size=3, strides=2, activation='relu', input_shape=(32, 32, 1))(input_conv)
    #convnet = Conv2D(16, kernel_size=3, strides=2, activation='relu')(convnet)
    convnet = MaxPooling2D(2, 2)(convnet)
    #convnet = Conv2D(32, kernel_size=3, strides=1, activation='relu')(convnet)
    convnet = Conv2D(16, kernel_size=3, strides=1, activation='relu')(convnet)
    convnet = MaxPooling2D(2, 2)(convnet)
    convnet = Flatten()(convnet)
    convnet = Dense(256, activation='relu')(convnet)
    convnet = Dropout(0.5)(convnet)
    convnet = Dense(256, activation='relu')(convnet)
    convnet = Dropout(0.5)(convnet)
    convnet = Dense(64, activation='relu')(convnet) # balance with densenet
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
    model = Dense(1, activation='linear')(model)
    model = Model(inputs=[convnet_left.input, convnet_right.input, densenet.input], outputs=model)
    return model

def train_model(model, data, is_x):
    (X_scalar, X_cnn_left, X_cnn_right, Y) = data
    print(X_cnn_left.shape)
    print(X_cnn_right)
    Y = [y[0 if is_x else 1] for y in Y]
    X_scalar_train, X_scalar_test, X_cnn_left_train, X_cnn_left_test, X_cnn_right_train, X_cnn_right_test, Y_train, Y_test = train_test_split(X_scalar, X_cnn_left, X_cnn_right, Y, test_size=0.25)
    X_cnn_left_train = np.expand_dims(X_cnn_left_train, axis=3)
    X_cnn_right_train = np.expand_dims(X_cnn_right_train, axis=3)
    X_cnn_left_test = np.expand_dims(X_cnn_left_test, axis=3)
    X_cnn_right_test = np.expand_dims(X_cnn_right_test, axis=3)
    opt = Adam(lr=1e-5, decay=1e-5 / 200)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', 'acc'])
    print("Successfully compiled model, begin training.")
    model.fit([X_cnn_left_train, X_cnn_right_train, X_scalar_train], Y_train,
                #validation_data=([X_cnn_left_test, X_cnn_right_test, X_scalar_test], Y_test),
                epochs=25, batch_size=256)
    print("Finished training, evaluating.")
    results = model.evaluate([X_cnn_left_test, X_cnn_right_test, X_scalar_test], Y_test, batch_size=64)
    print("Test loss, acc: ", results)
    return model
    

def main(preload_model=False):
    data = load_data()
    print("Training x network.")
    x_model = load_model('data/x_model.hdf5') if preload_model else build_model()
    x_model = train_model(x_model, data, True)
    print("Training y network.")
    y_model = load_model('data/y_model.hdf5') if preload_model else build_model()
    y_model = train_model(y_model, data, False)
    print("Dumping weights to disk.")
    x_model.save('data/x_model.hdf5')
    y_model.save('data/y_model.hdf5')

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main(input("Load weights [y/n] ") == 'y')