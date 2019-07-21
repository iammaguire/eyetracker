import numpy
import cv2
import sys
import os
from Xlib import display
from deepgaze.haar_cascade import haarCascade
from deepgaze.face_landmark_detection import faceLandmarkDetection

DEBUG = True

P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5,-5.0]) #45
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62

TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0,68))

def mouse_pos():
    data = display.Display().screen().root.query_pointer()._data
    return data["root_x"], data["root_y"]

def detect_eyes(gray_img, eye_cascade):
    left_eye = None
    right_eye = None
    crop = 5
    eyes = eye_cascade.detectMultiScale(gray_img, 1.1, 5)
    height = numpy.size(gray_img, 0)
    width = numpy.size(gray_img, 1)
    for (x, y, w, h) in eyes:
        if y + h > height / 2: # if detected eye is in bottom half of image
            pass
        center = x + w / 2
        if center < width * 0.5:
            left_eye = gray_img[y+crop:y+h-crop, x+crop:x+w-crop]
        else:
            right_eye = gray_img[y+crop:y+h-crop, x+crop:x+w-crop]
    return left_eye, right_eye

def main():
    video_capture = cv2.VideoCapture(0)
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))
    
    capturing = False
    captured_eye_data = []
    captured_head_data = []
    captured_mouse_data = []

    #Defining the camera matrix.
    #To have better result it is necessary to find the focal
    # lenght of the camera. fx/fy are the focal lengths (in pixels) 
    # and cx/cy are the optical centres. These values can be obtained 
    # roughly by approximation, for example in a 640x480 camera:
    # cx = 640/2 = 320
    # cy = 480/2 = 240
    # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
    f_y = f_x
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y], 
                                [0.0, 0.0, 1.0] ])
    camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
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
    face_cascade = haarCascade("/home/meet/Programming/python/eyetracker/haarcascade_frontalface_alt.xml", "/home/meet/Programming/python/eyetracker/haarcascade_profileface.xml")
    face_detector = faceLandmarkDetection('/home/meet/Programming/python/eyetracker/shape_predictor_68_face_landmarks.dat')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    no_face_counter = 0
    face_x1 = 0
    face_y1 = 0
    face_x2 = 0
    face_y2 = 0
    face_w = 0
    face_h = 0
    roi_x1 = 0
    roi_y1 = 0
    roi_x2 = cam_w
    roi_y2 = cam_h
    roi_w = cam_w
    roi_h = cam_h
    roi_resize_w = int(cam_w/10)
    roi_resize_h = int(cam_h/10)
    
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)

        # Return code: 1=Frontal, 2=FrontRotLeft, 
        # 3=FrontRotRight, 4=ProfileLeft, 5=ProfileRight.
        face_cascade.findFace(gray, True, True, True, True, 1.10, 1.10, 1.15, 1.15, 40, 40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=face_cascade.face_type)

        if face_cascade.face_type == 0: 
            no_face_counter += 1
        if no_face_counter == 50:
            no_face_counter = 0
            roi_x1 = 0
            roi_y1 = 0
            roi_x2 = cam_w
            roi_y2 = cam_h
            roi_w = cam_w
            roi_h = cam_h

        if face_cascade.face_type > 0:
            no_face_counter = 0
            #Because the dlib landmark detector wants a precise
            #boundary box of the face, it is necessary to resize
            #the box returned by the OpenCV haar detector.
            #Adjusting the frame for profile left
            if face_cascade.face_type == 4:
                face_margin_x1 = 20 - 10 #resize_rate + shift_rate
                face_margin_y1 = 20 + 5 #resize_rate + shift_rate
                face_margin_x2 = -20 - 10 #resize_rate + shift_rate
                face_margin_y2 = -20 + 5 #resize_rate + shift_rate
                face_margin_h = -0.7 #resize_factor
                face_margin_w = -0.7 #resize_factor
            #Adjusting the frame for profile right
            elif face_cascade.face_type == 5:
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
            face_x1 = face_cascade.face_x + roi_x1 + face_margin_x1
            face_y1 = face_cascade.face_y + roi_y1 + face_margin_y1
            face_x2 = face_cascade.face_x + face_cascade.face_w + roi_x1 + face_margin_x2
            face_y2 = face_cascade.face_y + face_cascade.face_h + roi_y1 + face_margin_y2
            face_w = face_cascade.face_w + int(face_cascade.face_w * face_margin_w)
            face_h = face_cascade.face_h + int(face_cascade.face_h * face_margin_h)
            roi_x1 = face_x1 - roi_resize_w
            if roi_x1 < 0: roi_x1 = 0
            roi_y1 = face_y1 - roi_resize_h
            if roi_y1 < 0: roi_y1 = 0
            roi_w = face_w + roi_resize_w + roi_resize_w
            if roi_w > cam_w: roi_w = cam_w
            roi_h = face_h + roi_resize_h + roi_resize_h
            if roi_h > cam_h: roi_h = cam_h    
            roi_x2 = face_x2 + roi_resize_w
            if roi_x2 > cam_w: roi_x2 = cam_w
            roi_y2 = face_y2 + roi_resize_h
            if roi_y2 > cam_h: roi_y2 = cam_h

            if DEBUG:
                text_x1 = face_x1
                text_y1 = face_y1 - 3
                if text_y1 < 0: text_y1 = 0
                cv2.putText(frame, "FACE", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
                cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)

            if face_cascade.face_type > 0:
                (mx, my) = mouse_pos()
                landmarks_2D = face_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2, points_to_return=TRACKED_POINTS)
                gray_face = cv2.cvtColor(frame[face_y1:face_y2, face_x1:face_x2], cv2.COLOR_BGR2GRAY)
                (left_eye, right_eye) = detect_eyes(gray_face, eye_cascade)
                    
                if DEBUG:
                    for point in landmarks_2D:
                        cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)
                #Applying the PnP solver to find the 3D pose
                # of the head from the 2D position of the
                # landmarks.
                #retval - bool
                #rvec - Output rotation vector that, together with tvec, brings 
                # points from the model coordinate system to the camera coordinate system.
                #tvec - Output translation vector.
                retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                                  landmarks_2D, 
                                                  camera_matrix, camera_distortion)

                #Now we project the 3D points into the image plane
                #Creating a 3-axis to be used as reference in the image.
                axis = numpy.float32([[50,0,0], 
                                      [0,50,0], 
                                      [0,0,50]])
                imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

                sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
                cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
                cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
                cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

                if left_eye is not None and right_eye is not None:
                    right_eye = cv2.resize(right_eye, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
                    left_eye = cv2.resize(left_eye, dsize=(32,32), interpolation=cv2.INTER_CUBIC)
                    cv2.imshow('left eye', left_eye)
                    cv2.imshow('right eye', right_eye)
                    if capturing:
                        captured_eye_data.append((left_eye, right_eye))
                        captured_head_data.append((rvec, tvec))
                        captured_mouse_data.append((mx, my))
                        cv2.circle(frame, (10, 10), 5, (0, 0, 255), -1)

        if DEBUG:
            text_x1 = roi_x1
            text_y1 = roi_y1 - 3
            if text_y1 < 0: text_y1 = 0
            cv2.putText(frame, "ROI", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1);
            cv2.rectangle(frame, 
                         (roi_x1, roi_y1), 
                         (roi_x2, roi_y2), 
                         (0, 255, 255),
                         2)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF
        capturing = key == ord('c')
        if key == ord('w'):
            num_existing_pairs = len(os.listdir('./data/imgs/')) / 2
            for idx, (left, right) in enumerate(captured_eye_data):
                cv2.imwrite('data/imgs/'+str(int(idx+num_existing_pairs))+'l.png', left)
                cv2.imwrite('data/imgs/'+str(int(idx+num_existing_pairs))+'r.png', right)
            with open('data/data.csv', 'a') as file:
                for idx, (mx, my) in enumerate(captured_mouse_data):
                    (rvec, tvec) = captured_head_data[idx]
                    file.write(str(rvec[0][0])+','+str(rvec[1][0])+','+str(rvec[2][0])+','+str(tvec[0][0])+','+str(tvec[1][0])+','+str(tvec[2][0])+','+str(mx)+','+str(my))
                    file.write('\n')
            captured_eye_data.clear()
        if key == ord('q'): break
   
    video_capture.release()

if __name__ == "__main__":
    main()
