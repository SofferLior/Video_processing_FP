import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats
import GeodisTK


#TODO: this is a copy from refrence
def choose_random_idx(mask,number_of_choices=200):
    idx = np.where(mask == 0)
    ran_choice = np.random.choice(len(idx[0]),number_of_choices)
    return np.column_stack((idx[0][ran_choice],idx[1][ran_choice]))


def calc_likelihood(rgb_image, ind, band_idx):
    idx = choose_random_idx(ind)

    value = rgb_image[idx[:,0], idx[:,1], :]
    kde = stats.gaussian_kde(value.T, bw_method=1)

    likelihood = kde(rgb_image[band_idx].T)

    return likelihood


def calc_distance_mag(yuv_image, trimap):
    temp_f_mask = trimap.copy()
    temp_f_mask[trimap < 255] = 0
    temp_f_mask[trimap == 255] = 1

    temp_b_mask = trimap.copy()
    temp_b_mask[trimap == 0] = 1
    temp_b_mask[trimap > 1] = 0

    foreground_distance_map = GeodisTK.geodesic2d_raster_scan(yuv_image, temp_f_mask, 1.0, 1)
    background_distance_map = GeodisTK.geodesic2d_raster_scan(yuv_image, temp_b_mask, 1.0, 1)

    return foreground_distance_map, background_distance_map


def find_small_area(trimap):
    expand_th = 50
    h, w = trimap.shape
    x, y, rect_w, rect_h = cv2.boundingRect(trimap)
    top_left_x = max(0, x - expand_th)
    top_left_y = max(0, y - expand_th)
    top_right_x = min(h, y + rect_h + expand_th)
    top_right_y = min(w, x + rect_w + expand_th)

    return top_left_x, top_left_y, top_right_x, top_right_y


def get_alpha(rgb_image,trimap, r=2):
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)
    top_left_x, top_left_y, top_right_x, top_right_y = find_small_area(trimap)

    cropped_trimap = trimap.copy()[top_left_y:top_right_x, top_left_x: top_right_y]
    # calculate distance map in a small area:
    foreground_distance_map, background_distance_map = calc_distance_mag(yuv_image[top_left_y:top_right_x, top_left_x: top_right_y], cropped_trimap)

    # find indices of the trimap band
    norm_f_dis_map = foreground_distance_map / (foreground_distance_map + background_distance_map)
    norm_b_dis_map = 1- norm_f_dis_map

    band = np.abs(norm_b_dis_map-norm_f_dis_map)
    band[band == 1] = 0
    band[band > 0] = 1
    if 0:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        new_im = trimap.copy()
        new_im[trimap == 255] = gray[trimap == 255]
        new_im[trimap == 0] = gray[trimap == 0]
        plt.imshow(new_im)

    band_idx = np.where(band == 1)

    decided_f_mask = (norm_f_dis_map < norm_b_dis_map-0.99).astype('uint8')
    decided_b_mask = (norm_b_dis_map>= norm_f_dis_map - 0.99).astype('uint8')

    likelihood_map_f = calc_likelihood(yuv_image[top_left_y:top_right_x, top_left_x: top_right_y], decided_f_mask, band_idx)
    likelihood_map_b = calc_likelihood(yuv_image[top_left_y:top_right_x, top_left_x: top_right_y], decided_b_mask, band_idx)

    w_f = np.multiply(np.power(foreground_distance_map[band_idx], -r), likelihood_map_f)
    w_b = np.multiply(np.power(background_distance_map[band_idx], -r), likelihood_map_b)
    temp_alpha = 1000*np.multiply(w_f, np.power(w_f + w_b, -1))

    alpha = trimap.copy()
    alpha[top_left_y:top_right_x, top_left_x: top_right_y][band_idx] = temp_alpha

    return alpha


def get_trimap(image, size, erosion=False):
    #TODO: change this code, this is not mine
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [4]: a binary image (black & white only), name of the image, dilation pixels
                the last argument is optional; i.e., how many iterations will the image get eroded
    Output    : a trimap
    """
    row = image.shape[0]
    col = image.shape[1]
    pixels = 2*size + 1      ## Double and plus 1 to have an odd-sized kernel
    kernel = np.ones((pixels,pixels),np.uint8)   ## Pixel of extension I get

    if erosion is not False:
        erosion = int(erosion)
        erosion_kernel = np.ones((3,3), np.uint8)                     ## Design an odd-sized erosion kernel
        image = cv2.erode(image, erosion_kernel, iterations=10)  ## How many erosion do you expect
        image = np.where(image > 0, 255, image)                       ## Any gray-clored pixel becomes white (smoothing)
        # Error-handler to prevent entire foreground annihilation
        if cv2.countNonZero(image) == 0:
            print("ERROR: foreground has been entirely eroded")

    dilation = cv2.dilate(image, kernel, iterations=5)

    dilation = np.where(dilation == 255, 127, dilation) 	## WHITE to GRAY
    remake = np.where(dilation != 127, 0, dilation)		## Smoothing
    remake = np.where(image > 127, 200, dilation)		## mark the tumor inside GRAY

    remake = np.where(remake < 127, 0, remake)		## Embelishment
    remake = np.where(remake > 200, 0, remake)		## Embelishment
    remake = np.where(remake == 200, 255, remake)		## GRAY to WHITE

    for i in range(0,row):
        for j in range (0,col):
            if (remake[i,j] != 0 and remake[i,j] != 255):
                remake[i,j] = 127

    return remake


def video_matting(input_path, binary_path, new_background_path, video_data):
    cap = cv2.VideoCapture(input_path)
    binary_cap = cv2.VideoCapture(binary_path)
    out_tracked = cv2.VideoWriter(os.path.join('Output', 'matting.avi'), video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']))
    new_background_image = cv2.imread(new_background_path)

    curr_frame = 0
    while cap.isOpened():
        ret, cur_frame_rgb = cap.read()
        binary_ret, cur_binary_frame = binary_cap.read()
        if (ret or binary_ret) is False:
            break
        curr_frame += 1
        print(f'frame {curr_frame}')

        cur_frame_bin_grey = cv2.cvtColor(cur_binary_frame, cv2.COLOR_BGR2GRAY)
        cur_frame_yuv_image = cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2YUV)

        # Create Trimap
        trimap = get_trimap(cur_frame_bin_grey, 5, erosion=5)
        if 1:
            gray = cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2GRAY)
            new_im = trimap.copy()
            new_im[trimap == 255] = gray[trimap == 255]
            new_im[trimap == 0] = gray[trimap == 0]
            plt.imshow(new_im)
        print('Calculated Trimap')

        # Create Alpha maps
        alpha_map = get_alpha(cur_frame_rgb, trimap)
        print('Calculated Alpha map')

        alpha = alpha_map.astype(float)/255

        alpha_rgb = np.zeros_like(cur_frame_rgb)
        for i in range(0, 3):
            alpha_rgb[:, :, i] = alpha[:]

        f = np.multiply(alpha_rgb, cur_frame_rgb.astype(float))
        b = np.multiply(1.0-alpha_rgb, new_background_image.astype(float))

        frame_matted = cv2.add(f, b).astype('uint8')

        if curr_frame%10 == 0:
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(cur_frame_rgb)
            plt.subplot(1,3,2)
            plt.imshow(cur_binary_frame)
            plt.subplot(1,3,3)
            plt.imshow(frame_matted)
            plt.show()
            print(f'figure for frame number: {curr_frame}')

        out_tracked.write(frame_matted)

    out_tracked.release()
    return cap