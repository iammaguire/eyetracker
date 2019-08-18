import cv2
import numpy as np
from keras.models import load_model

class BlinkDetector:
    def __init__(self):
        self.IMG_SIZE = (34, 26)
        self.model = load_model('data/blink_model.h5')
    
    def detect(self, l, r):
        input_l = cv2.resize(l.copy(), (self.IMG_SIZE[0], self.IMG_SIZE[1])).astype(np.float32) / 255.0
        cv2.imshow('l', input_l)
        input_l = np.expand_dims(np.expand_dims(input_l, axis=0), axis=3)
        input_r = cv2.resize(r.copy(), (self.IMG_SIZE[0], self.IMG_SIZE[1])).astype(np.float32) / 255.0
        cv2.imshow('r', input_r)
        input_r = np.expand_dims(np.expand_dims(input_r, axis=0), axis=3)
        pred_l = self.model.predict(input_l)
        pred_r = self.model.predict(input_r)
        return pred_l, pred_r