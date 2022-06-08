import cv2
import numpy as np
from scipy import stats
import GeodisTK


def calc_kde(rgb_image, idx):
    rand_idx_choice = np.random.choice(len(idx[0]), 200)
    sampled_idx = np.column_stack((idx[0][rand_idx_choice],idx[1][rand_idx_choice]))
    sampled_image = rgb_image[sampled_idx[:,0], sampled_idx[:,1], :]
    kde = stats.gaussian_kde(sampled_image.T, bw_method=1)
    return kde


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

    band = norm_b_dis_map-norm_f_dis_map
    background_idx= np.where(band < -0.99)
    foreground_idx = np.where(band > 0.99)

    temp_band = np.zeros_like(band)
    temp_band[foreground_idx] = 1
    temp_band[background_idx] = -1
    band_idx = np.where(temp_band == 0)

    kde_f = calc_kde(rgb_image[top_left_y:top_right_x, top_left_x: top_right_y], foreground_idx)
    kde_b = calc_kde(rgb_image[top_left_y:top_right_x, top_left_x: top_right_y], background_idx)

    w_f = np.multiply(np.power(foreground_distance_map[band_idx], -r), kde_f(rgb_image[top_left_y:top_right_x, top_left_x: top_right_y][band_idx].T))
    w_b = np.multiply(np.power(background_distance_map[band_idx], -r), kde_b(rgb_image[top_left_y:top_right_x, top_left_x: top_right_y][band_idx].T))
    temp_alpha = np.multiply(w_f, np.power(w_f + w_b, -1))

    cropped_alpha = np.zeros_like(cropped_trimap).astype(np.float)
    cropped_alpha[foreground_idx] = 1
    cropped_alpha[band_idx] = temp_alpha

    alpha = trimap.copy().astype(np.float)
    alpha[top_left_y:top_right_x, top_left_x: top_right_y] = cropped_alpha

    return alpha


def get_trimap(gray_image):
    dilation_kernel = np.ones((11, 11), np.uint8)
    erosion_kernel = np.ones((3, 3), np.uint8)

    eroded_image = cv2.erode(gray_image, erosion_kernel, iterations=10)
    eroded_image[np.where(eroded_image > 0)] = 255

    dilation_image = cv2.dilate(eroded_image, dilation_kernel, iterations=5)
    dilation_image[np.where(dilation_image == 255)] = 127

    trimap = np.where(eroded_image > 127, 200, dilation_image)
    trimap[np.where(trimap < 127)] = 0
    trimap[np.where(trimap > 200)] = 0
    trimap[np.where(trimap == 200)] = 255

    for i in range(0, gray_image.shape[0]):
        for j in range(0, gray_image.shape[1]):
            if trimap[i, j] != 0 and trimap[i, j] != 255:
                trimap[i, j] = 127

    return trimap


def create_alpha(input_path, binary_path, out_alpha_path, video_data):
    cap = cv2.VideoCapture(input_path)
    binary_cap = cv2.VideoCapture(binary_path)
    out_alpha = cv2.VideoWriter(out_alpha_path, video_data['fourcc'], video_data['fps'],
                                  (video_data['w'], video_data['h']))

    curr_frame = 0
    while cap.isOpened():
        ret, cur_frame_rgb = cap.read()
        binary_ret, cur_binary_frame = binary_cap.read()
        if (ret or binary_ret) is False:
            break
        curr_frame += 1

        cur_frame_bin_grey = cv2.cvtColor(cur_binary_frame, cv2.COLOR_BGR2GRAY)

        # Create Trimap
        trimap = get_trimap(cur_frame_bin_grey)

        # Create Alpha maps
        alpha_map = get_alpha(cur_frame_rgb, trimap)

        alpha_rgb = np.zeros_like(cur_frame_rgb).astype(float)
        for i in range(0, 3):
            alpha_rgb[:, :, i] = alpha_map[:]

        out_alpha.write((255*alpha_rgb).astype('uint8'))

        if np.mod(curr_frame, 20) == 0:
            print(f'Alpha - frame num {curr_frame} / {video_data["frames_num"]}')

    out_alpha.release()


def video_matting(input_path, alpha_path, new_background_path, out_matted_path, video_data):
    cap = cv2.VideoCapture(input_path)
    alpha_cap = cv2.VideoCapture(alpha_path)
    out_matted = cv2.VideoWriter(out_matted_path, video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']))

    # resize background image in case needed
    new_background_image_original = cv2.imread(new_background_path)
    new_background_image = cv2.resize(new_background_image_original, (video_data['w'], video_data['h']))

    curr_frame = 0
    while cap.isOpened():
        ret, cur_frame_rgb = cap.read()
        alpha_ret, cur_alpha_frame = alpha_cap.read()
        if (ret or alpha_ret) is False:
            break
        curr_frame += 1

        norm_alpha_frame = cur_alpha_frame.astype(float) / 255

        f = np.multiply(norm_alpha_frame, cur_frame_rgb)
        b = np.multiply(1.0 - norm_alpha_frame, new_background_image)

        frame_matted = cv2.add(f, b).astype('uint8')

        out_matted.write(frame_matted)

    out_matted.release()
