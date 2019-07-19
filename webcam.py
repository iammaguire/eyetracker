import cv2
import numpy as np
import tensorflow as tf
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from deepgaze.haar_cascade import haarCascade

face_cascade = haarCascade("/home/meet/Programming/python/eyetracker/haarcascade_frontalface_alt.xml", "/home/meet/Programming/python/eyetracker/haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cam = cv2.VideoCapture(0)
cam_w = int(cam.get(3))
cam_h = int(cam.get(4))
no_face_counter = 0
face_x0 = 0
face_y0 = 0
face_x1 = 0
face_y1 = 0
face_w = 0
face_h = 0
roi_x0 = 0
roi_y0 = 0
roi_x1 = cam_w
roi_y1 = cam_h
roi_w = cam_w
roi_h = cam_h
roi_resize_w = int(cam_w / 10)
roi_resize_h = int(cam_h / 10)

def detect_eyes(img, gray_img):
    left_eye = None
    right_eye = None
    eyes = eye_cascade.detectMultiScale(gray_img, 1.3, 5)
    height = np.size(img, 0)
    width = np.size(img, 1)
    for (x, y, w, h) in eyes:
        if y + h > height / 2: # if detected eye is in bottom half of image
            pass
        center = x + w / 2
        if center < width * 0.5:
            left_eye = gray_img[y:y + h, x:x + w]
        else:
            right_eye = gray_img[y:y + h, x:x + w]
    return left_eye, right_eye

def detect_face(img, gray_img):
    face = None
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = img[y:y + h, x:x + w]
    return face

def update(cam):
    ret_val, img = cam.read()
    img = cv2.flip(img, 1)
    cv2.imshow('webcam', img)
    gray_img = cv2.cvtColor(img[roi_y0:ri_y1, roi_x0:roi_x1], cv2.COLOR_BGR2GRAY)
    face_cascade.findFace(gray_img, True, True, True, True, 1.10, 1.10, 1.15, 1.15, 40, 40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=face_cascade.face_type)
    
    if face_cascade/face_type == 0:
        no_face_counter++
    if no_face_counter >= 50:
        no_face_counter = 0
        roi_x0 = 0
        roi_y0 = 0
        roi_x1 = cam_w
        roi_y1 = cam_h
        roi_w = cam_w
        roi_h = cam_h
    
    #face = detect_face(img, gray_img)
    if face is not None:
        cv2.imshow('face', face)
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        (left_eye, right_eye) = detect_eyes(face, gray_face)
        if left_eye is not None and right_eye is not None:
            cv2.imshow('left eye', left_eye)
            cv2.imshow('right eye', right_eye)

def show_webcam():
    while True:
        update(cam)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam()

if __name__ == '__main__':
    main()
