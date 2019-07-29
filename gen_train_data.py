import cv2
import sys
import os
import numpy as np
from Xlib import display
from face_tracker import FaceTracker

DEBUG = True

def mouse_pos():
    data = display.Display().screen().root.query_pointer()._data
    return data["root_x"], data["root_y"]

def add_gaussian_noise(X_imgs): # augmentation
    gaussian_noise_imgs = []
    row, col = X_imgs[0].shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for X_img in X_imgs:
        X_img = X_img.astype(np.float32)
        gaussian = np.random.random((row, col)).astype(np.float32)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs

def main():
    tracker = FaceTracker()    
    capturing = False
    captured_eye_data = []
    captured_head_data = []
    captured_mouse_data = []
    
    while True:    
        frame = tracker.tick()
        if DEBUG:
            text_x1 = tracker.face_x1
            text_y1 = tracker.face_y1 - 3
            if text_y1 < 0: text_y1 = 0
            cv2.putText(frame, "FACE", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
            cv2.rectangle(frame, (tracker.face_x1, tracker.face_y1), (tracker.face_x2, tracker.face_y2), (0, 255, 0), 2)
        if tracker.face_type > 0:
            if DEBUG:
                for point in tracker.landmarks_2D:
                    cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)
            (mx, my) = mouse_pos()
            axis = np.float32([[50,0,0], [0,50,0], [0,0,50]])
            imgpts, jac = cv2.projectPoints(axis, tracker.rvec, tracker.tvec, tracker.camera_matrix, tracker.camera_distortion)
            sellion_xy = (tracker.landmarks_2D[7][0], tracker.landmarks_2D[7][1])
            cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
            cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED
            if tracker.left_eye is not None and tracker.right_eye is not None:
                cv2.imshow('left eye', tracker.left_eye)
                cv2.imshow('right eye', tracker.right_eye)
                if capturing:
                    captured_eye_data.append((tracker.left_eye, tracker.right_eye))
                    captured_head_data.append((tracker.rvec, tracker.tvec))
                    captured_mouse_data.append((mx, my))
                    cv2.circle(frame, (10, 10), 5, (0, 0, 255), -1)
        if DEBUG:
            text_x1 = tracker.roi_x1
            text_y1 = tracker.roi_y1 - 3
            if text_y1 < 0: text_y1 = 0
            cv2.putText(frame, "ROI", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1);
            cv2.rectangle(frame, 
                         (tracker.roi_x1, tracker.roi_y1), 
                         (tracker.roi_x2, tracker.roi_y2), 
                         (0, 255, 255), 2)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF
        capturing = key == ord('c')
        if key == ord('w'):
            num_existing_pairs = len(os.listdir('./data/imgs/')) / 2
            for idx, (left, right) in enumerate(captured_eye_data):
                (left_gauss, right_gauss) = add_gaussian_noise([left, right])
                cv2.imwrite('data/imgs/l/'+str(int((idx*2+1)+num_existing_pairs))+'.png', left_gauss)
                cv2.imwrite('data/imgs/r/'+str(int((idx*2+1)+num_existing_pairs))+'.png', right_gauss)
                cv2.imwrite('data/imgs/l/'+str(int((idx*2)+num_existing_pairs))+'.png', left)
                cv2.imwrite('data/imgs/r/'+str(int((idx*2)+num_existing_pairs))+'.png', right)
            with open('data/data.csv', 'a') as file:
                for idx, (mx, my) in enumerate(captured_mouse_data):
                    (rvec, tvec) = captured_head_data[idx]
                    file.write(str(rvec[0][0])+','+str(rvec[1][0])+','+str(rvec[2][0])+','+str(tvec[0][0])+','+str(tvec[1][0])+','+str(tvec[2][0])+','+str(mx)+','+str(my))
                    file.write('\n')
                    file.write(str(rvec[0][0])+','+str(rvec[1][0])+','+str(rvec[2][0])+','+str(tvec[0][0])+','+str(tvec[1][0])+','+str(tvec[2][0])+','+str(mx)+','+str(my)) # write twice due to gauss augmentation
                    file.write('\n')
            print("Wrote " + str(len(captured_eye_data ) * 2) + " data points to disk.")
            captured_eye_data.clear()
            captured_mouse_data.clear()
            captured_head_data.clear()
        if key == ord('q'): break

if __name__ == "__main__":
    main()


# 1763r.png <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,