import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# TODO: delete all debugs
DEBUG = False


def background_subtraction_first_frames(cap):

    output_first_binary_frames = []
    output_first_fg_frames = []
    backSub = cv2.createBackgroundSubtractorKNN()
    input_frames_for_bs = []

    for i in range(100):
        ret, cur_frame_rgb = cap.read()
        input_frames_for_bs.append(cur_frame_rgb)

    for ii, frame in enumerate(input_frames_for_bs[::-1]):

        fg_mask = backSub.apply(frame)
        fg_mask_processed = post_process_fg(fg_mask)
        fg_mask_processed = np.stack([fg_mask_processed, fg_mask_processed, fg_mask_processed], axis=-1)
        fg_frame = np.multiply(fg_mask_processed / 255, frame).astype(np.uint8)
        output_first_fg_frames.append(fg_frame)
        output_first_binary_frames.append(fg_mask_processed)

        if DEBUG and np.mod(ii, 5) == 0:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.subplot(2, 2, 2)
            plt.title('fg binary')
            plt.imshow(fg_mask, cmap='gray')
            plt.subplot(2, 2, 3)
            plt.title('fg_mask_opened_1')
            plt.imshow(fg_mask_processed, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title('fg frame')
            plt.imshow(cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB))

    return output_first_binary_frames[::-1], output_first_fg_frames[::-1]


def background_subtraction(cap, video_data, out_extracted_fg_path, out_binary_path, first_binary_frames, first_fg_frames):

    out_extracted_fg = cv2.VideoWriter(out_extracted_fg_path, video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)
    out_binary = cv2.VideoWriter(out_binary_path, video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']), True)

    backSub = cv2.createBackgroundSubtractorKNN()

    curr_frame = 0

    while (cap.isOpened()):
        if np.mod(curr_frame, 20) == 0:
            print(f'BS - frame num {curr_frame} / {video_data["frames_num"]}')
        curr_frame += 1
        ret, cur_frame_rgb = cap.read()
        if ret is False:
            break

        fg_mask = backSub.apply(cur_frame_rgb)
        fg_mask_processed = post_process_fg(fg_mask)

        fg_mask_processed = np.stack([fg_mask_processed, fg_mask_processed, fg_mask_processed], axis=-1)
        fg_frame = np.multiply(fg_mask_processed / 255, cur_frame_rgb).astype(np.uint8)
        if curr_frame < 20:
            out_binary.write(first_binary_frames[curr_frame])
            out_extracted_fg.write(first_fg_frames[curr_frame])
            continue
        out_binary.write(fg_mask_processed.astype(np.uint8))
        out_extracted_fg.write(fg_frame)

        if DEBUG and np.mod(curr_frame, 20) == 0:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2RGB))
            plt.subplot(2, 2, 2)
            plt.title('fg binary')
            plt.imshow(fg_mask, cmap='gray')
            plt.subplot(2, 2, 3)
            plt.title('fg_mask_opened_1')
            plt.imshow(fg_mask_processed, cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title('fg frame')
            plt.imshow(cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB))

    out_binary.release()
    out_extracted_fg.release()

    if DEBUG:
        plt.show(block=False)

    return out_extracted_fg


def post_process_fg(fg_mask_gray):

    a, fg_binary = cv2.threshold(fg_mask_gray, 200, 255, cv2.THRESH_BINARY)
    kernel_5 = np.ones((5, 5), np.uint8)
    kernel_8 = np.ones((8, 8), np.uint8)
    kernel_10 = np.ones((10, 10), np.uint8)
    kernel_12 = np.ones((12, 12), np.uint8)
    fg_mask_opened_1 = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, kernel_5)
    fg_mask_closed_1 = cv2.morphologyEx(fg_mask_opened_1, cv2.MORPH_CLOSE, kernel_10)
    fg_mask_opened_2 = cv2.morphologyEx(fg_mask_closed_1, cv2.MORPH_OPEN, kernel_8)
    fg_mask_closed_2 = cv2.morphologyEx(fg_mask_opened_2, cv2.MORPH_CLOSE, kernel_12)

    con_com = cv2.connectedComponentsWithStats(fg_mask_closed_2, 8, cv2.CV_32S)
    largest_component = [0, 0, 0, 0]
    largest_size = 0
    for i in range(1, con_com[0]):
        area = con_com[2][i, cv2.CC_STAT_AREA]
        if area > largest_size:
            x = con_com[2][i, cv2.CC_STAT_LEFT]
            y = con_com[2][i, cv2.CC_STAT_TOP]
            w = con_com[2][i, cv2.CC_STAT_WIDTH]
            h = con_com[2][i, cv2.CC_STAT_HEIGHT]
            largest_component = [x, y, w, h]
            largest_size = area
    fg_mask_closed_2[:largest_component[1], :] = 0
    fg_mask_closed_2[largest_component[1] + largest_component[3]:, :] = 0
    fg_mask_closed_2[:, :largest_component[0]] = 0
    fg_mask_closed_2[:, largest_component[0] + largest_component[2]:] = 0

    return fg_mask_closed_2
