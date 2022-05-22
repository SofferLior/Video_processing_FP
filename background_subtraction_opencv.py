import os
import cv2
import matplotlib.pyplot as plt

DEBUG = True


def background_subtraction(cap, video_data, output_path, time_window_size=70, subtraction_th=70, algo='MOG2'):

    out_extracted_fg = cv2.VideoWriter(os.path.join(output_path, 'foreground.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)
    out_bg_only = cv2.VideoWriter(os.path.join(output_path, 'background.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)
    out_binary = cv2.VideoWriter(os.path.join(output_path, 'binary_foreground.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)

    if algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
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

        if DEBUG:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(cur_frame_rgb)
            plt.subplot(1, 2, 2)
            plt.imshow(fgMask)

        a=0
