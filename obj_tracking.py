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
    # start_point = (0,0)
    # end_point = (100,100)

    curr_frame = 0
    while (cap.isOpened()):
        ret, cur_frame_rgb = cap.read()
        binary_ret, cur_binary_frame = binary_cap.read()
        if (ret or binary_ret) is False:
            break
        curr_frame += 1
        # start_point = (start_point[0] + 1, start_point[1] + 1)
        # end_point = (end_point[0] + 1, end_point[1] + 1)
        grey_binary = cv2.cvtColor(cur_binary_frame, cv2.COLOR_RGB2GRAY)
        x, y, w, h = cv2.boundingRect(grey_binary)
        if (x==0 or y==0):
            print('Issue with the frame - blank')
        out_tracked.write(draw_rectangle(cur_frame_rgb, x, y, w, h))

    out_tracked.release()
