import cv2
import numpy as np
from deepgaze.haar_cascade import haarCascade
from deepgaze.face_landmark_detection import faceLandmarkDetection

P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = np.float32([-20.0, 65.5,-5.0]) #45
P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62

TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68))

class FaceTracker:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.cam_w = int(self.video_capture.get(3))
        self.cam_h = int(self.video_capture.get(4))
        self.blinked = False  
        #Defining the camera matrix.
        #To have better result it is necessary to find the focal
        # lenght of the camera. fx/fy are the focal lengths (in pixels) 
        # and cx/cy are the optical centres. These values can be obtained 
        # roughly by approximation, for example in a 640x480 camera:
        # cx = 640/2 = 320
        # cy = 480/2 = 240
        # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
        self.c_x = self.cam_w / 2
        self.c_y = self.cam_h / 2
        self.f_x = self.c_x / np.tan(60/2 * np.pi / 180)
        self.f_y = self.f_x
        self.camera_matrix = np.float32([[self.f_x, 0.0, self.c_x],
                                    [0.0, self.f_y, self.c_y], 
                                    [0.0, 0.0, 1.0] ])
        self.camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
        self.landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                    P3D_GONION_RIGHT,
                                    P3D_MENTON,
                                    P3D_GONION_LEFT,
                                    P3D_LEFT_SIDE,
                                    P3D_FRONTAL_BREADTH_RIGHT,
                                    P3D_FRONTAL_BREADTH_LEFT,
                                    P3D_SELLION,
                                    P3D_NOSE,
                                    P3D_SUB_NOSE,
                                    P3D_RIGHT_EYE,
                                    P3D_RIGHT_TEAR,
                                    P3D_LEFT_TEAR,
                                    P3D_LEFT_EYE,
                                    P3D_STOMION])
        self.face_cascade = haarCascade("/home/meet/Programming/python/eyetracker/haarcascade_frontalface_alt.xml", "/home/meet/Programming/python/eyetracker/haarcascade_profileface.xml")
        self.face_detector = faceLandmarkDetection('/home/meet/Programming/python/eyetracker/shape_predictor_68_face_landmarks.dat')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.no_face_counter = 0
        self.face_x1 = 0
        self.face_y1 = 0
        self.face_x2 = 0
        self.face_y2 = 0
        self.face_w = 0
        self.face_h = 0
        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = self.cam_w
        self.roi_y2 = self.cam_h
        self.roi_w = self.cam_w
        self.roi_h = self.cam_h
        self.roi_resize_w = int(self.cam_w/10)
        self.roi_resize_h = int(self.cam_h/10)
        self.face_type = -1
        self.no_eye_counter = 0
    
    def tick(self):
        ret, frame = self.video_capture.read()
        gray = cv2.cvtColor(frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2], cv2.COLOR_BGR2GRAY)

        # Return code: 1=Frontal, 2=FrontRotLeft, 
        # 3=FrontRotRight, 4=ProfileLeft, 5=ProfileRight.
        self.face_cascade.findFace(gray, True, True, True, True, 1.10, 1.10, 1.15, 1.15, 40, 40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=self.face_type)
        self.face_type = self.face_cascade.face_type
        if self.face_type == 0: 
            self.no_face_counter += 1
        if self.no_face_counter == 50:
            self.no_face_counter = 0
            self.roi_x1 = 0
            self.roi_y1 = 0
            self.roi_x2 = self.cam_w
            self.roi_y2 = self.cam_h
            self.roi_w = self.cam_w
            self.roi_h = self.cam_h

        if self.face_type > 0:
            self.no_face_counter = 0
            #Because the dlib landmark detector wants a precise
            #boundary box of the face, it is necessary to resize
            #the box returned by the OpenCV haar detector.
            #Adjusting the frame for profile left
            if self.face_type == 4:
                face_margin_x1 = 20 - 10 #resize_rate + shift_rate
                face_margin_y1 = 20 + 5 #resize_rate + shift_rate
                face_margin_x2 = -20 - 10 #resize_rate + shift_rate
                face_margin_y2 = -20 + 5 #resize_rate + shift_rate
                face_margin_h = -0.7 #resize_factor
                face_margin_w = -0.7 #resize_factor
            #Adjusting the frame for profile right
            elif self.face_type == 5:
                face_margin_x1 = 20 + 10
                face_margin_y1 = 20 + 5
                face_margin_x2 = -20 + 10
                face_margin_y2 = -20 + 5
                face_margin_h = -0.7
                face_margin_w = -0.7
            #No adjustments
            else:
                face_margin_x1 = 0
                face_margin_y1 = 0
                face_margin_x2 = 0
                face_margin_y2 = 0
                face_margin_h = 0
                face_margin_w = 0
            self.face_x1 = self.face_cascade.face_x + self.roi_x1 + face_margin_x1
            self.face_y1 = self.face_cascade.face_y + self.roi_y1 + face_margin_y1
            self.face_x2 = self.face_cascade.face_x + self.face_cascade.face_w + self.roi_x1 + face_margin_x2
            self.face_y2 = self.face_cascade.face_y + self.face_cascade.face_h + self.roi_y1 + face_margin_y2
            self.face_w = self.face_cascade.face_w + int(self.face_cascade.face_w * face_margin_w)
            self.face_h = self.face_cascade.face_h + int(self.face_cascade.face_h * face_margin_h)
            self.roi_x1 = self.face_x1 - self.roi_resize_w
            if self.roi_x1 < 0: self.roi_x1 = 0
            self.roi_y1 = self.face_y1 - self.roi_resize_h
            if self.roi_y1 < 0: self.roi_y1 = 0
            self.roi_w = self.face_w + self.roi_resize_w + self.roi_resize_w
            if self.roi_w > self.cam_w: self.roi_w = self.cam_w
            self.roi_h = self.face_h + self.roi_resize_h + self.roi_resize_h
            if self.roi_h > self.cam_h: self.roi_h = self.cam_h    
            self.roi_x2 = self.face_x2 + self.roi_resize_w
            if self.roi_x2 > self.cam_w: self.roi_x2 = self.cam_w
            self.roi_y2 = self.face_y2 + self.roi_resize_h
            if self.roi_y2 > self.cam_h: self.roi_y2 = self.cam_h
            
            self.landmarks_2D = self.face_detector.returnLandmarks(frame, self.face_x1, self.face_y1, self.face_x2, self.face_y2, points_to_return=TRACKED_POINTS)
            gray_face = cv2.cvtColor(frame[self.face_y1:self.face_y2, self.face_x1:self.face_x2], cv2.COLOR_BGR2GRAY)
            (self.left_eye, self.right_eye) = self.detect_eyes(gray_face)
            if self.left_eye is not None and self.right_eye is not None:
                self.right_eye = cv2.resize(self.right_eye, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
                self.left_eye = cv2.resize(self.left_eye, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
            else:
                self.no_eye_counter += 1
            if self.no_eye_counter >= 25:
                self.blinked = True
                self.no_eye_counter = 0
            else:
                self.blinked = False
            #Applying the PnP solver to find the 3D pose
            # of the head from the 2D position of the
            # landmarks.
            #retval - bool
            #rvec - Output rotation vector that, together with tvec, brings 
            # points from the model coordinate system to the camera coordinate system.
            #tvec - Output translation vector.
            retval, self.rvec, self.tvec = cv2.solvePnP(self.landmarks_3D, 
                                            self.landmarks_2D, 
                                            self.camera_matrix, self.camera_distortion)
        return frame

    def detect_eyes(self, gray_img):
        left_eye = None
        right_eye = None
        crop = 5
        eyes = self.eye_cascade.detectMultiScale(gray_img, 1.07, 5)
        height = np.size(gray_img, 0)
        width = np.size(gray_img, 1)
        for (x, y, w, h) in eyes:
            if y + h > height / 1.75: # if detected eye is in bottom half of image
                continue
            center = x + w / 2
            if center < width * 0.5:
                left_eye = gray_img[y+crop:y+h-crop, x+crop:x+w-crop]
            else:
                right_eye = gray_img[y+crop:y+h-crop, x+crop:x+w-crop]
        return left_eye, right_eye

    def __del__(self):
        self.video_capture.release()