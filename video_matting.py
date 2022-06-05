import cv2
import os
import numpy as np
import wdt
from scipy import stats
import matplotlib.pyplot as plt

def simple_matt_frame(frame_RGB, new_BG, Alpha_map):
    # simple matting without optimizing background pixel search
    AlphaRGB = np.zeros_like(frame_RGB)
    for i in range(0, 3):
        AlphaRGB[:, :, i] = Alpha_map[:]
    matted_frame = np.multiply(frame_RGB, AlphaRGB) + np.multiply(new_BG, (1-AlphaRGB))
    return matted_frame


def get_Alpha_map(norm_dist_map_F, norm_dist_map_B, Prob_map_F, Prob_map_B, Trimap, r):
    # input: distance maps, probability maps, mask for border area, r (power)
    # Output: Alpha map for border area

    # calculate new distance maps using Trimap

    # calculate weights according to formula given in class
    # TODO: turn 0.01 to parameter epsilon, in order to avoid inf values
    WF = np.multiply(np.power(norm_dist_map_F+0.01, -2), Prob_map_F)
    WB = np.multiply(np.power(norm_dist_map_B+0.01, -2), Prob_map_B)
    DEN = np.power(WF + WB, (-1))
    band_alpha = np.multiply(WF,DEN)
    Alpha_map = np.zeros_like(Trimap)
    Alpha_map[:] = Trimap
    Alpha_map[Trimap == 5] = band_alpha[Trimap == 5]

    return Alpha_map


def get_Trimap(dist_map_FG,dist_map_BG,dilation):
    #OutPut: trimap with

    #       Background 0

    #       Foreground 1

    #       Unknown 5

    # Normalize distances
    #dist_map_FG *= 255.0 / dist_map_FG.max()
    #dist_map_BG *= 255.0 / dist_map_BG.max()

    temp_Trimap = np.zeros_like(dist_map_FG)
    dist_diff = dist_map_BG[:] - dist_map_FG[:]
    temp_Trimap[dist_diff>5e16] = 5
    pixels = 2 * dilation + 1
    kernel = np.ones((pixels, pixels), np.uint8)

    temp_Trimap = cv2.erode(temp_Trimap, kernel, iterations=1)
    Trimap = cv2.dilate(temp_Trimap, kernel, iterations=2)
    Trimap = Trimap - temp_Trimap
    Trimap[temp_Trimap==5] = 1
    Trimap = cv2.dilate(Trimap, kernel, iterations=1)

    return Trimap


def get_CDF(img_rgb, omega):
    # TODO: consider moving num_of_bins to parameters
    # TODO: how to include info from all channels to CDF?
    num_of_bins = 256
    height, width, channels = img_rgb.shape

    imgR = img_rgb[:, :, 0]
    imgG = img_rgb[:, :, 1]
    imgB = img_rgb[:, :, 2]

    maskR = imgR[np.nonzero(omega)]
    maskG = imgG[np.nonzero(omega)]
    maskB = imgB[np.nonzero(omega)]

    kdeR = stats.gaussian_kde(maskR)
    kdeG = stats.gaussian_kde(maskG)
    kdeB = stats.gaussian_kde(maskB)

    binR = np.linspace(maskR.min(), maskR.max(), num_of_bins)
    binG = np.linspace(maskG.min(), maskG.max(), num_of_bins)
    binB = np.linspace(maskB.min(), maskB.max(), num_of_bins)

    densityR = kdeR(binR)
    densityG = kdeG(binG)
    densityB = kdeB(binB)

    CalculatedCDF_vec = densityR[np.reshape(imgR, (1, height*width))]*densityG[np.reshape(imgG, (1, height*width))]*densityB[np.reshape(imgB, (1, height*width))]
    CalculatedCDF_mat = np.reshape(CalculatedCDF_vec, (height, width))

    # TODO: Change CDF matrix creation to more efficient method
    # TotalCDF = np.zeros((num_of_bins, num_of_bins, num_of_bins), dtype=float)
    # for i in range(num_of_bins):
    #    for j in range(num_of_bins):
    #        for k in range(num_of_bins):
    #            TotalCDF[i, j, k] = densityR[i]*densityG[j]*densityB[k]

    return CalculatedCDF_mat


def trimap(image, size, erosion=False):
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
        image = cv2.erode(image, erosion_kernel, iterations=erosion)  ## How many erosion do you expect
        image = np.where(image > 0, 255, image)                       ## Any gray-clored pixel becomes white (smoothing)
        # Error-handler to prevent entire foreground annihilation
        if cv2.countNonZero(image) == 0:
            print("ERROR: foreground has been entirely eroded")

    dilation = cv2.dilate(image, kernel, iterations=10)

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

        if curr_frame > 4:
            cur_frame_bin_grey = cv2.cvtColor(cur_binary_frame, cv2.COLOR_BGR2GRAY)

            '''            omega_fg_eroded = cv2.erode(cur_frame_bin_grey, np.ones((5, 5), np.uint8), iterations=10)
                        OmegaBG = np.zeros_like(cur_frame_bin_grey)
                        OmegaBG[cur_frame_bin_grey > 0] = 5
                        OmegaBG[cur_frame_bin_grey == 0] = 255
                        OmegaBG[cur_frame_bin_grey == 5] = 0
                        omega_bg_eroded = cv2.erode(OmegaBG, np.ones((15, 15), np.uint8), iterations=10)
            '''

            '''            plt.figure()
                        plt.subplot(1,5,1)
                        plt.imshow(cur_frame_bin_grey)
                        plt.subplot(1,5,2)
                        plt.imshow(omega_fg_eroded)
                        plt.subplot(1,5,3)
                        plt.imshow(OmegaBG)
                        plt.subplot(1,5,4)
                        plt.imshow(omega_bg_eroded)
                        plt.subplot(1,5,5)
                        plt.imshow(cv2.dilate(cur_frame_bin_grey, np.ones((5, 5), np.uint8), iterations=10))
                        a=0'''
            '''            # create eroded image - TODO: optimize erosion filter according to input binary image
                        ret_binary, cur_frame_binarized = cv2.threshold(cur_frame_bin_grey, 127, 255, cv2.THRESH_BINARY)
                        OmegaFG = cur_frame_binarized
                        OmegaFG_Eroded = cv2.erode(OmegaFG, np.ones((5, 5), np.uint8), iterations=1)
                        OmegaBG = np.zeros_like(cur_frame_binarized)
                        OmegaBG[cur_frame_binarized > 0] = 5
                        OmegaBG[cur_frame_binarized == 0] = 255
                        OmegaBG[cur_frame_binarized == 5] = 0
            
                        OmegaBG_Eroded = cv2.erode(OmegaBG, np.ones((15, 15), np.uint8), iterations=10)
            
                        # calculate CDF using KDE
                        CDF_given_F = get_CDF(cur_frame_rgb, OmegaFG_Eroded)
                        CDF_given_B = get_CDF(cur_frame_rgb, OmegaBG_Eroded)
            
                        # Create F/B likelyhood map
                        Prob_map_F = np.zeros_like(cur_frame_rgb[:, :, 1])
                        Prob_map_B = np.zeros_like(cur_frame_rgb[:, :, 1])
                        Prob_map_F = np.divide(CDF_given_F, (CDF_given_F + CDF_given_B))
                        Prob_map_B = np.divide(CDF_given_B, (CDF_given_F + CDF_given_B))
            
                        Prob_map_F = np.uint8(Prob_map_F > 0.5) * 255
                        Prob_map_B = np.uint8(Prob_map_B > 0.5) * 255
            
                        # progress
                        print('Calculated likelyhood')
            
                        # Create normalized maps of gradient of likelyhood
                        Prob_map_F_grad = cv2.Laplacian(Prob_map_F, cv2.CV_16S)
                        Prob_map_B_grad = cv2.Laplacian(Prob_map_B, cv2.CV_16S)
            
                        abs_Prob_map_F_grad = np.abs(Prob_map_F_grad)
                        normalized_pr_pixel_Prob_map_F_grad = np.uint8(
                            255 * np.divide(abs_Prob_map_F_grad, np.max(abs_Prob_map_F_grad)))
                        abs_Prob_map_B_grad = np.abs(Prob_map_B_grad)
                        normalized_pr_pixel_Prob_map_B_grad = np.uint8(
                            255 * np.divide(abs_Prob_map_B_grad, np.max(abs_Prob_map_B_grad)))
            
                        ## calculate cost field for F/B
                        cost_field_F = wdt.map_image_to_costs(normalized_pr_pixel_Prob_map_F_grad, OmegaFG_Eroded)
                        distance_transform_F = wdt.get_weighted_distance_transform(cost_field_F)
            
                        # create BG distance map by inversing FG distmap - because of bug in WDT
                        max_cost_F = np.max(cost_field_F)
                        cost_field_B = max_cost_F * np.ones_like(cost_field_F) - cost_field_F
                        distance_transform_B = wdt.get_weighted_distance_transform(cost_field_B)
'''
            # Create Trimap
            Trimap = trimap(cur_frame_bin_grey,5,erosion=10) #get_Trimap(distance_transform_F, distance_transform_B, 5)
            print('Calculated Trimap')

            # Create Alpha maps
            #Alpha_map = get_Alpha_map(distance_transform_F, distance_transform_B, Prob_map_F, Prob_map_B, Trimap, 2)
            alpha_map = cv2.alphamat.infoFlow(cur_frame_rgb,Trimap)

            print('Calculated Alpha map')

            # Create new frame with background
            ## resize background image
            new_BG_resized = np.resize(new_background_image, (video_data['h'], video_data['w'], 3))

            # frame_matted = matt_frame(cur_frame_orig, new_BG_resized, Alpha_map, Trimap_radius_rho)
            frame_matted = simple_matt_frame(cur_frame_rgb, new_BG_resized, alpha_map)

            # Save frame in output video
            frame_out = frame_matted

            out_tracked.write(frame_out)
        else:
            out_tracked.write(cur_frame_rgb)
    out_tracked.release()
    return cap