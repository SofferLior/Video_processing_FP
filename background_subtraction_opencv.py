import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

DEBUG = True


def background_subtraction(cap, video_data, output_path, time_window_size=70, subtraction_th=70, algo='knn'):

    out_extracted_fg = cv2.VideoWriter(os.path.join(output_path, 'foreground.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)
    out_bg_only = cv2.VideoWriter(os.path.join(output_path, 'background.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)
    out_binary = cv2.VideoWriter(os.path.join(output_path, 'binary_foreground.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)

    if algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=30)
    else:
        backSub = cv2.createBackgroundSubtractorKNN()

    curr_frame = 0

    while (cap.isOpened()):
        print(f'BS - frame num {curr_frame} / {video_data["frames_num"]}')
        curr_frame += 1
        ret, cur_frame_rgb = cap.read()
        if ret is False:
            break

        fgMask = backSub.apply(cur_frame_rgb)
        a, fg_binary = cv2.threshold(fgMask,200,255,cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask_opened = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, kernel)
        fg_mask_opened = cv2.morphologyEx(fg_mask_opened, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((10, 10), np.uint8)
        fg_mask_closed = cv2.morphologyEx(fg_mask_opened, cv2.MORPH_CLOSE, kernel)
        fg_mask_closed = cv2.morphologyEx(fg_mask_closed, cv2.MORPH_CLOSE, kernel)

        if DEBUG and np.mod(curr_frame, 30) == 0:
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2RGB))
            plt.subplot(2, 3, 2)
            plt.imshow(fgMask, cmap='gray')
            plt.subplot(2, 3, 3)
            plt.imshow(fg_binary, cmap='gray')
            plt.subplot(2, 3, 4)
            plt.imshow(fg_mask_opened, cmap='gray')
            plt.subplot(2, 3, 5)
            plt.imshow(fg_mask_closed, cmap='gray')
            plt.subplot(2, 3, 6)
            # todo: arrange this
            plt.imshow(np.multiply(np.stack([fg_mask_closed / 255, fg_mask_closed / 255, fg_mask_closed / 255], axis=-1), cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2RGB)))


    plt.show(block=False)
    a=0
