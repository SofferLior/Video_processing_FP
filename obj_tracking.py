import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_rectangle(frame, x, y, w, h):
    color = (0, 128, 0)  # green
    thickness = 3
    return cv2.rectangle(frame,(x, y), (x+w, y+h), color,thickness)


def track_obj(input_path, binary_path, video_data):
    cap = cv2.VideoCapture(input_path)
    binary_cap = cv2.VideoCapture(binary_path)
    out_tracked = cv2.VideoWriter(os.path.join('Output', 'extracted.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']))
    start_point = (0,0)
    end_point = (100,100)

    curr_frame = 0
    while (cap.isOpened()):
        ret, cur_frame_rgb = cap.read()
        binary_ret, cur_binary_frame = binary_cap.read()
        if (ret or binary_ret) is False:
            break
        curr_frame += 1
        start_point = (start_point[0] + 1, start_point[1] + 1)
        end_point = (end_point[0] + 1, end_point[1] + 1)
        grey_binary = cv2.cvtColor(cur_binary_frame, cv2.COLOR_RGB2GRAY)
        contours, h = cv2.findContours(grey_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = 0
        idx = 0
        for i, contour in enumerate(contours):
            cont_area = cv2.contourArea(contour)
            if cont_area > max_area:
                max_area = cont_area
                idx = i
        max_contour = contours[idx]
        x, y, w, h = cv2.boundingRect(max_contour)

        if True and np.mod(curr_frame, 30) == 0:
            plt.imshow(cv2.drawContours(cur_binary_frame,[max_contour], 0, (0, 255, 0), 3))
            plt.show(block=False)
            print(f'cur frame: {curr_frame}')
        out_tracked.write(draw_rectangle(cur_frame_rgb, x, y, w, h))

    out_tracked.release()