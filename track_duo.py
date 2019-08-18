import cv2
import keras
import pyautogui
import numpy as np
import tensorflow as tf
from collections import deque
from keras import backend as K
from keras.models import load_model
from train_tracker import load_data
from face_tracker import FaceTracker
from detect_blink import BlinkDetector

def main():
    x_model = load_model('data/x_model.hdf5')
    y_model = load_model('data/y_model.hdf5')
    X_scalar, Y = load_data(False)
    normaliser = np.amax(X_scalar, axis=0)
    tracker = FaceTracker()
    blink_detector = BlinkDetector()
    clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
    last_x = deque([None]*3)
    last_y = deque([None]*3)
    while True:    
        frame = tracker.tick()
        if tracker.face_type > 0:
            if tracker.blinked:
                pyautogui.click()
            if tracker.left_eye is not None and tracker.right_eye is not None and tracker.rvec is not None and tracker.tvec is not None:
                (blink_l, blink_r) = blink_detector.detect(tracker.left_eye, tracker.right_eye)
                print(str(blink_l) + " : " + str(blink_r))
            
                X_scalar = np.concatenate((tracker.rvec, tracker.tvec)).flatten() / normaliser
                left = np.expand_dims(tracker.left_eye, axis=3)  / 255.0
                right = np.expand_dims(tracker.right_eye, axis=3)  / 255.0
                x_pred = x_model.predict([[left], [right], [X_scalar]])
                y_pred = y_model.predict([[left], [right], [X_scalar]])
                x = clamp(x_pred[0]*1920, 0, 1920)
                y = clamp(y_pred[0]*1080, 0, 1080)
                last_x[next((i for i, v in enumerate(last_x) if v is None), -1) if None in last_x else 0] = x
                last_y[next((i for i, v in enumerate(last_y) if v is None), -1) if None in last_y else 0] = y
                if None not in last_x: last_x.rotate(1)
                if None not in last_y: last_y.rotate(1)
                if x_pred is not None and y_pred is not None and None not in last_x and None not in last_y:
                    pyautogui.moveTo(sum(last_x) / len(last_x), sum(last_y) / len(last_y))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == "__main__":
    config = tf.ConfigProto(
        device_count={'GPU': 1},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True
    )

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    session = tf.Session(config=config)

    keras.backend.set_session(session)
    with session.as_default():
            with session.graph.as_default():
                main()