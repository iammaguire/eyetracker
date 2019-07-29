import cv2
import keras
import pyautogui
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from train_tracker import load_data
from face_tracker import FaceTracker

def main():
    x_model = load_model('data/x_model.hdf5')
    y_model = load_model('data/y_model.hdf5')
    X_scalar, Y = load_data(False)
    normaliser = np.amax(X_scalar, axis=0)
    tracker = FaceTracker()
    clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
    while True:    
        frame = tracker.tick()
        if tracker.face_type > 0:
            if tracker.left_eye is not None and tracker.right_eye is not None and tracker.rvec is not None and tracker.tvec is not None:
                X_scalar = np.concatenate((tracker.rvec, tracker.tvec)).flatten() / normaliser
                left = np.expand_dims(tracker.left_eye, axis=3)  / 255.0
                right = np.expand_dims(tracker.right_eye, axis=3)  / 255.0
                x_pred = x_model.predict([[left], [right], [X_scalar]])
                y_pred = y_model.predict([[left], [right], [X_scalar]])
                x = clamp(x_pred[0]*1920, 0, 1920)
                y = clamp(y_pred[0]*1080, 0, 1080)
                if x_pred is not None and y_pred is not None:
                    pyautogui.moveTo(x, y)
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