# TODO:
#  1. extract the walking person - extracted.avi
#  2. binary mask of the walking person - binary.avi
import os
from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt


def substract_single_image(cur_original_image_rgb, cur_median_window_frame_rgb, threshold, debug=True):
    cur_original_image_gray = cv2.cvtColor(cur_original_image_rgb, cv2.COLOR_BGR2GRAY)
    cur_median_window_frame_gray = cv2.cvtColor(cur_median_window_frame_rgb, cv2.COLOR_BGR2GRAY)

    blur_k_size=5
    cur_original_image_gray_blur = cv2.blur(cur_original_image_gray,(blur_k_size,blur_k_size))
    cur_median_window_frame_gray_blur = cv2.blur(cur_median_window_frame_gray,(blur_k_size,blur_k_size))
    # cur_original_image_gray_blur = cv2.blur(cur_original_image_gray_blur,(blur_k_size,blur_k_size))
    # cur_median_window_frame_gray_blur = cv2.blur(cur_median_window_frame_gray_blur,(blur_k_size,blur_k_size))

    foreground_mask_binary = np.uint8(np.abs(np.int16(cur_original_image_gray_blur) - np.int16(cur_median_window_frame_gray_blur)) > threshold)*255

    # processing the binary image
    foreground_mask_binary = cv2.medianBlur(foreground_mask_binary, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask_binary = cv2.erode(foreground_mask_binary, kernel, iterations=3)
    foreground_mask_binary = cv2.medianBlur(foreground_mask_binary, 5)
    foreground_mask_binary = cv2.dilate(foreground_mask_binary, kernel, iterations=7)

    foreground_mask_binary = foreground_mask_binary > 150
    idx = (foreground_mask_binary == 1)
    cur_extracted_fg_frame_rgb = np.zeros(np.shape(cur_original_image_rgb))
    cur_extracted_fg_frame_rgb[idx] = cur_original_image_rgb[idx]

    cur_extracted_fg_frame_rgb = np.uint8(cur_extracted_fg_frame_rgb)
    foreground_mask_binary = np.uint8(foreground_mask_binary)*255

    # todo: delete this
    if debug:
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(cur_original_image_gray_blur, cmap='gray')
        plt.title('cur_original_image_gray_blur')
        plt.subplot(2,2,2)
        plt.imshow(cur_median_window_frame_gray_blur, cmap='gray')
        plt.title('cur_median_window_frame_gray_blur')
        plt.subplot(2,2,3)
        plt.imshow(foreground_mask_binary, cmap='gray')
        plt.title('foreground_mask_binary')
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(cur_extracted_fg_frame_rgb, cv2.COLOR_BGR2RGB))
        plt.title('cur_extracted_fg_frame_rgb')
        plt.show(block=False)

    return [foreground_mask_binary, cur_extracted_fg_frame_rgb]


def background_subtraction(cap, video_data, output_path, time_window_size=70, subtraction_th=70):

    out_extracted_fg = cv2.VideoWriter(os.path.join(output_path, 'foreground.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)
    out_bg_only = cv2.VideoWriter(os.path.join(output_path, 'background.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)
    out_binary = cv2.VideoWriter(os.path.join(output_path, 'binary_foreground.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)

    image_rgb_window = deque([np.zeros([video_data['h'], video_data['w'], 3], dtype='uint8')]*time_window_size)
    red_window = deque([np.zeros([video_data['h'], video_data['w']])]*time_window_size)
    green_window = deque([np.zeros([video_data['h'], video_data['w']])]*time_window_size)
    blue_window = deque([np.zeros([video_data['h'], video_data['w']])]*time_window_size)

    curr_frame = 0

    while (cap.isOpened()):
        print(f'BS - frame num {curr_frame} / {video_data["frames_num"]}')
        curr_frame += 1
        ret, cur_frame_rgb = cap.read()
        if ret is False:
            break

        if curr_frame < time_window_size:
            continue

        median_window_frame_red = np.median(np.array(red_window), axis=0)
        median_window_frame_green = np.median(np.array(green_window), axis=0)
        median_window_frame_blue = np.median(np.array(blue_window), axis=0)

        cur_median_window_frame_rgb = np.uint8(np.zeros([video_data['h'], video_data['w'], 3]))
        cur_median_window_frame_rgb[:, :, 0] = np.uint8(median_window_frame_blue)
        cur_median_window_frame_rgb[:, :, 1] = np.uint8(median_window_frame_green)
        cur_median_window_frame_rgb[:, :, 2] = np.uint8(median_window_frame_red)

        if curr_frame == time_window_size:
            for cur_original_image in image_rgb_window:
                cur_binary_fg_mask, cur_extracted_fg_frame_rgb = substract_single_image(cur_original_image, cur_median_window_frame_rgb, subtraction_th, debug=False)
                out_extracted_fg.write(cur_extracted_fg_frame_rgb)
                out_binary.write(cur_binary_fg_mask)
                out_bg_only.write(cur_median_window_frame_rgb)
            continue

        cur_binary_fg_mask, cur_extracted_fg_frame_rgb = substract_single_image(cur_frame_rgb, cur_median_window_frame_rgb, threshold=100, debug=True)
        out_extracted_fg.write(cur_extracted_fg_frame_rgb)
        out_binary.write(cur_binary_fg_mask)
        out_bg_only.write(cur_median_window_frame_rgb)

    return cap