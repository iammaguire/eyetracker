import cv2
import sys
import os
import time
import random
import numpy as np
from Xlib import display
from face_tracker import FaceTracker
from gen_train_data import add_gaussian_noise

DEBUG = True

def main():
    tracker = FaceTracker()    
    capturing = False
    captured_eye_data = []
    captured_head_data = []
    captured_mouse_data = []
    aoi_size = 512
    aoi_x = random.randrange(0, 1920-aoi_size)
    aoi_y = random.randrange(0, 1080-aoi_size)
    follow_x = random.randrange(aoi_x, aoi_x + aoi_size)
    follow_y = random.randrange(aoi_y, aoi_y + aoi_size)
    follow_dx = 2.2
    follow_dy = 0.4
    pause = True
    first = True

    while True:
        follow_x += follow_dx
        follow_y += follow_dy
        
        img = np.ones((1080, 1920, 3), np.uint8) * 255
        if follow_x > aoi_x + aoi_size or follow_x < aoi_x or follow_y > aoi_y + aoi_size or follow_y < aoi_y:
            aoi_x = random.randrange(0, 1920-aoi_size)
            aoi_y = random.randrange(0, 1080-aoi_size)
            follow_x = random.randrange(aoi_x, aoi_x + aoi_size)
            follow_y = random.randrange(aoi_y, aoi_y + aoi_size)
            x_cond = follow_x > aoi_x + aoi_size / 2
            y_cond = follow_y > aoi_y + aoi_size / 2
            follow_dx = random.randrange(-9.0, -1.0) if x_cond else random.randrange(1.0, 9.0)
            follow_dy = random.randrange(-9.0, -1.0) if y_cond else random.randrange(1.0, 9.0)
            cv2.circle(img, (int(follow_x), int(follow_y)), 12, (0, 255, 0), -1)
            cv2.line(img, (int(follow_x), int(follow_y)), (int(follow_x + follow_dx * 10), int(follow_y + follow_dy * 10)), (0, 255, 0), 1)
            pause = True
        cv2.circle(img, (int(follow_x), int(follow_y)), 8, (0,0,255), -1)
        cv2.rectangle(img, (aoi_x, aoi_y), (aoi_x + aoi_size, aoi_y + aoi_size), (0,255,0), 2)
        cv2.imshow('trainer', img)
        cv2.setWindowProperty('trainer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    
        cv2.moveWindow('trainer', 0, 0)
        
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
            (mx, my) = (follow_x, follow_y)
            axis = np.float32([[50,0,0], [0,50,0], [0,0,50]])
            imgpts, jac = cv2.projectPoints(axis, tracker.rvec, tracker.tvec, tracker.camera_matrix, tracker.camera_distortion)
            sellion_xy = (tracker.landmarks_2D[7][0], tracker.landmarks_2D[7][1])
            cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
            cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
            cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED
            if tracker.left_eye is not None and tracker.right_eye is not None:
                cv2.imshow('left eye', tracker.left_eye)
                cv2.imshow('right eye', tracker.right_eye)
                cv2.moveWindow('left eye', 2225, 0)
                cv2.moveWindow('right eye', 2580, 0)
                captured_eye_data.append((tracker.left_eye, tracker.right_eye))
                captured_head_data.append((tracker.rvec, tracker.tvec))
                captured_mouse_data.append((mx, my))
                cv2.circle(frame, (10, 10), 5, (0, 0, 255), -1)
                first = False
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
        cv2.moveWindow('Video', 2980, 0)
        #if pause:
        #    pause = False
        #    time.sleep(0.5)
        cv2.waitKey(1)
        if pause:
            pause = False
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
                elif key == ord('w'):
                    num_existing_pairs = len(os.listdir('./data/imgs/r'))
                    if len(captured_mouse_data) != len(captured_head_data) or len(captured_mouse_data) != len(captured_eye_data):
                        print("Differing array sizes.")
                        sys.exit(0)
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
                    sys.exit(0)
                elif key == ord('q'):
                    sys.exit(0)

if __name__ == "__main__":
    main()

