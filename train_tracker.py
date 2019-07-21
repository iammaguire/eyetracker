import os
import cv2
import sys
import pandas
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, concatenate
from sklearn.model_selection import train_test_split

def quit_with(err):
    print(err)
    sys.exit()

def load_data():
    frame = pandas.read_csv("data/data.csv", header=None)
    data = frame.values
    X_scalar = data[:, 0:6]
    Y = data[:, 6:]
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
            
    print(np.array(X_cnn_right)[0])
    return X_scalar, X_cnn_left, X_cnn_right, Y

def build_convnet(input_conv):
    convnet = Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 1))(input_conv)
    convnet = Conv2D(32, kernel_size=3, activation='relu')(convnet)
    convnet = MaxPooling2D(pool_size=(2, 2))(convnet)
    convnet = Conv2D(64, kernel_size=3, activation='relu')(convnet)
    convnet = Flatten()(convnet)
    convnet = Dense(512, activation='relu')(convnet)
    convnet = Dropout(0.5)(convnet)
    convnet = Dense(32, activation='relu')(convnet) # balance with densenet
    return Model(inputs=input_conv, outputs=convnet)

def build_densenet(input_dense):
    densenet = Dense(12, activation='relu')(input_dense)
    densenet = Dense(8, activation='relu')(densenet)
    return Model(inputs=input_dense, outputs=densenet)

def build_model():
    input_conv_left = Input(shape=(32, 32, 1))
    input_conv_right = Input(shape=(32, 32, 1))
    input_dense = Input(shape=(6,))
    convnet_left = build_convnet(input_conv_left)
    convnet_right = build_convnet(input_conv_right)
    densenet = build_densenet(input_dense)
    model = concatenate([convnet_left.output, convnet_right.output, densenet.output])
    model = Dense(4, activation='relu')(model)
    model = Dense(2, activation='linear')(model)
    model = Model(inputs=[convnet_left.input, convnet_right.input, densenet.input], outputs=model)
    return model

def main():
    X_scalar, X_cnn_left, X_cnn_right, Y = load_data()
    model = build_model()
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='mean_absolute_percentage_error', optimizer=opt)
    print("Successfully compiled model.")
    

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    main()