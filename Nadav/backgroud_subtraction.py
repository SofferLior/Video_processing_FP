import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from collections import deque

def substract_single_image_rgb(cur_original_image_rgb, cur_median_window_frame_rgb, threshold):
    cur_original_image_r = cur_original_image_rgb[:, :, 2]
    cur_original_image_g = cur_original_image_rgb[:, :, 1]
    cur_original_image_b = cur_original_image_rgb[:, :, 0]
    cur_median_window_frame_r = cur_median_window_frame_rgb[:, :, 2]
    cur_median_window_frame_g = cur_median_window_frame_rgb[:, :, 1]
    cur_median_window_frame_b = cur_median_window_frame_rgb[:, :, 0]

    cur_original_image_blur_r = cv2.blur(cur_original_image_r,(5,5))
    cur_original_image_blur_g = cv2.blur(cur_original_image_g,(5,5))
    cur_original_image_blur_b = cv2.blur(cur_original_image_b,(5,5))
    cur_median_window_frame_gray_r = cv2.blur(cur_median_window_frame_r,(5,5))
    cur_median_window_frame_gray_g = cv2.blur(cur_median_window_frame_g,(5,5))
    cur_median_window_frame_gray_b = cv2.blur(cur_median_window_frame_b,(5,5))

    subtracted_r = np.abs(np.int16(cur_original_image_blur_r) - np.int16(cur_median_window_frame_gray_r))
    subtracted_g = np.abs(np.int16(cur_original_image_blur_g) - np.int16(cur_median_window_frame_gray_g))
    subtracted_b = np.abs(np.int16(cur_original_image_blur_b) - np.int16(cur_median_window_frame_gray_b))

    sub_of_subtracted = subtracted_r + subtracted_g + subtracted_b

    threshold, foreground_mask_binary = cv2.threshold(sub_of_subtracted, threshold, 255, cv2.THRESH_BINARY)

    # foreground_mask_binary_r = np.abs(np.int16(cur_original_image_blur_r) - np.int16(cur_median_window_frame_gray_r)) > threshold
    # foreground_mask_binary_g = np.abs(np.int16(cur_original_image_blur_g) - np.int16(cur_median_window_frame_gray_g)) > threshold
    # foreground_mask_binary_b = np.abs(np.int16(cur_original_image_blur_b) - np.int16(cur_median_window_frame_gray_b)) > threshold
    # foreground_mask_binary = np.multiply(np.multiply(foreground_mask_binary_r, foreground_mask_binary_g), foreground_mask_binary_b)

    foreground_mask_binary = np.uint8(foreground_mask_binary)

    # processing the binary image
    foreground_mask_binary = cv2.medianBlur(foreground_mask_binary, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask_binary = cv2.erode(foreground_mask_binary, kernel, iterations=1)
    foreground_mask_binary = cv2.dilate(foreground_mask_binary, kernel, iterations=1)
    foreground_mask_binary = cv2.dilate(foreground_mask_binary, kernel, iterations=1)
    foreground_mask_binary = cv2.erode(foreground_mask_binary, kernel, iterations=1)
    foreground_mask_binary = foreground_mask_binary > 150

    idx = (foreground_mask_binary == 1)
    cur_extracted_fg_frame_rgb = np.zeros(np.shape(cur_original_image_rgb))
    cur_extracted_fg_frame_rgb[idx] = cur_original_image_rgb[idx]

    cur_extracted_fg_frame_rgb = np.uint8(cur_extracted_fg_frame_rgb)
    foreground_mask_binary = np.uint8(foreground_mask_binary)*255

    if 0:
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(cv2.cvtColor(cur_original_image_rgb,cv2.COLOR_BGR2RGB))
        plt.title('cur_original_image_rgb')
        plt.subplot(2,2,2)
        plt.imshow(cv2.cvtColor(cur_median_window_frame_rgb,cv2.COLOR_BGR2RGB))
        plt.title('cur_median_window_frame_rgb')
        plt.subplot(2,2,3)
        plt.imshow(foreground_mask_binary)
        plt.title('foreground_mask_binary')
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(cur_extracted_fg_frame_rgb,cv2.COLOR_BGR2RGB))
        plt.title('cur_extracted_fg_frame_rgb')

    return [foreground_mask_binary, cur_extracted_fg_frame_rgb]

def substract_single_image(cur_original_image_rgb, cur_median_window_frame_rgb, threshold):
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

    if 0:
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(cur_original_image_gray_blur)
        plt.title('cur_original_image_gray_blur')
        plt.subplot(2,2,2)
        plt.imshow(cur_median_window_frame_gray_blur)
        plt.title('cur_median_window_frame_gray_blur')
        plt.subplot(2,2,3)
        plt.imshow(foreground_mask_binary)
        plt.title('foreground_mask_binary')
        plt.subplot(2, 2, 4)
        plt.imshow(cur_extracted_fg_frame_rgb)
        plt.title('cur_extracted_fg_frame_rgb')

    return [foreground_mask_binary, cur_extracted_fg_frame_rgb]

def background_subtraction(gui, input_video_full_path, output_video_path, time_window_size = 70, subtraction_th=70):
    output_extracted_video_full_path = os.path.join(output_video_path, 'extracted.avi')
    output_binary_video_full_path = os.path.join(output_video_path, 'binary.avi')
    output_bg_video_full_path = os.path.join(output_video_path, 'bg.avi')

    cap = cv2.VideoCapture(input_video_full_path)
    if cap.isOpened() is False:
        print('Error openning video stream or file')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out_extracted_fg = cv2.VideoWriter(output_extracted_video_full_path, fourcc, fps, (frame_width, frame_height), True)
    out_bg_only = cv2.VideoWriter(output_bg_video_full_path, fourcc, fps, (frame_width, frame_height), True)
    out_binary = cv2.VideoWriter(output_binary_video_full_path, fourcc, fps, (frame_width, frame_height), False)

    image_rgb_window = deque([np.zeros([frame_height, frame_width, 3])]*time_window_size)
    red_window = deque([np.zeros([frame_height, frame_width])]*time_window_size)
    green_window = deque([np.zeros([frame_height, frame_width])]*time_window_size)
    blue_window = deque([np.zeros([frame_height, frame_width])]*time_window_size)

    frame_number = 0
    gui.progress["maximum"] = num_of_frames
    gui.progress["value"] = 0
    gui.pb_precentage_label.config(text='0%')
    gui.window.update()
    while (cap.isOpened()):
        gui.progress["value"] = frame_number
        cur_precentage = float('%.2f' % ((frame_number/num_of_frames)*100))
        gui.pb_precentage_label.config(text=str(cur_precentage)+'%')
        gui.window.update()
        print(f'bs - frame num {frame_number}:{num_of_frames}')
        frame_number += 1
        ret, cur_frame_rgb = cap.read()
        if ret is False:
            break
        image_rgb_window.append(cur_frame_rgb)
        red_window.append(cur_frame_rgb[:, :, 2])
        green_window.append(cur_frame_rgb[:, :, 1])
        blue_window.append(cur_frame_rgb[:, :, 0])
        image_rgb_window.popleft()
        red_window.popleft()
        green_window.popleft()
        blue_window.popleft()

        if frame_number < time_window_size:
            continue

        median_window_frame_red = np.median(np.array(red_window), axis=0)
        median_window_frame_green = np.median(np.array(green_window), axis=0)
        median_window_frame_blue = np.median(np.array(blue_window), axis=0)

        cur_median_window_frame_rgb = np.uint8(np.zeros([frame_height, frame_width, 3]))
        cur_median_window_frame_rgb[:, :, 0] = np.uint8(median_window_frame_blue)
        cur_median_window_frame_rgb[:, :, 1] = np.uint8(median_window_frame_green)
        cur_median_window_frame_rgb[:, :, 2] = np.uint8(median_window_frame_red)

        if 0:
            cv2.imshow("only BG", cur_median_window_frame_rgb)
            cv2.waitKey(10)

        if frame_number == time_window_size:
            for cur_original_image in image_rgb_window:
                cur_binary_fg_mask, cur_extracted_fg_frame_rgb = substract_single_image(cur_original_image, cur_median_window_frame_rgb, subtraction_th)
                out_extracted_fg.write(cur_extracted_fg_frame_rgb)
                out_binary.write(cur_binary_fg_mask)
                out_bg_only.write(cur_median_window_frame_rgb)
            continue

        cur_binary_fg_mask, cur_extracted_fg_frame_rgb = substract_single_image(cur_frame_rgb, cur_median_window_frame_rgb, subtraction_th)
        out_extracted_fg.write(cur_extracted_fg_frame_rgb)
        out_binary.write(cur_binary_fg_mask)
        out_bg_only.write(cur_median_window_frame_rgb)
