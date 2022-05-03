import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from collections import deque
from scipy import ndimage
from scipy import stats
import wdt

#def matt_frame(cur_frame, new_BG, Alpha_map, Trimap, dilation):
#    # dilation should be x2 of trimap radius
#    pixels = dilation*2+1
#    NarrBand = Tripmap[Trimap > 1]
#    pixels_in_NarrBand = NarrBand.len


#    matted_frame = np.zeros_like(cur_frame)
#    # take definite pixels from BG and FG
#    for i in range (0,3):
#        matted_frame[(Alpha_map == 1),i] = cur_frame [(Alpha_map == 1),i]
#        matted_frame[(Alpha_map == 0),i] = new_BG [(Alpha_map == 0),i]

#    # look around every pixel in narrow band to find optimal combination
#    for j in range(0,pixels_in_NarrBand):
#        cost = 1000
#        for y in range(NarrBand[0,j]-pixels,NarrBand[0,j]+pixels):
#            for x in range(NarrBand[1,j]-pixels,NarrBand[1,j]+pixels):
#                for c in range(0,3):
#                    diff[] = (Alpha_map[NarrBand[0,j],NarrBand[1,j]] * cur_frame[y,x])
#                if (Alp)



#    return matted_frame


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

    if(0):
        plt.figure()
        plt.subplot(221), plt.imshow(Trimap), plt.title('Trimap')
        plt.subplot(222), plt.imshow(Alpha_map), plt.title('Alpha_map')
        plt.subplot(223), plt.imshow(WF), plt.title('WF')
        plt.subplot(224), plt.imshow(WB), plt.title('WB')

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

    if(0):
       plt.figure()
       plt.subplot(331), plt.imshow(dist_map_FG), plt.title('dist_map_FG image')
       plt.subplot(332), plt.imshow(dist_map_BG), plt.title('dist_map_BG image')
       plt.subplot(333), plt.imshow(dist_diff), plt.title('dist_diff')
       plt.subplot(334), plt.imshow(temp_Trimap), plt.title('temp_Trimap')
       plt.subplot(335), plt.imshow(Trimap), plt.title('Trimap')
       plt.show()

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

    if(0):
        plt.figure()
        plt.plot(densityR)
        plt.figure()
        plt.plot(densityG)
        plt.figure()
        plt.plot(densityB)

    CalculatedCDF_vec = densityR[np.reshape(imgR, (1, height*width))]*densityG[np.reshape(imgG, (1, height*width))]*densityB[np.reshape(imgB, (1, height*width))]
    CalculatedCDF_mat = np.reshape(CalculatedCDF_vec, (height, width))

    # TODO: Change CDF matrix creation to more efficient method
    # TotalCDF = np.zeros((num_of_bins, num_of_bins, num_of_bins), dtype=float)
    # for i in range(num_of_bins):
    #    for j in range(num_of_bins):
    #        for k in range(num_of_bins):
    #            TotalCDF[i, j, k] = densityR[i]*densityG[j]*densityB[k]

    return CalculatedCDF_mat


def video_matting(gui, input_video_path, input_video_extracted_path, output_video_path, input_video_binary_path, New_Background_path, Trimap_radius_rho, power_r):
    # Paths and parameters
    output_video_full_path = os.path.join(output_video_path, 'matted.avi')
    New_BG = cv2.imread(New_Background_path)

    # Open Extracted Video
    cap_extracted = cv2.VideoCapture(input_video_extracted_path)
    if cap_extracted.isOpened() is False:
        print('Error openning video stream or file')

    frame_width = int(cap_extracted.get(3))
    frame_height = int(cap_extracted.get(4))
    num_of_frames = int(cap_extracted.get(cv2.CAP_PROP_FRAME_COUNT))
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = cap_extracted.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = cap_extracted.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    # Open stabilized Video
    cap = cv2.VideoCapture(input_video_path)
    if cap.isOpened() is False:
        print('Error opening video stream or file')

    # Open binary video
    cap_bin = cv2.VideoCapture(input_video_binary_path)
    if cap_bin.isOpened() is False:
        print('Error opening video stream or file')


    # Set up output video
    out = cv2.VideoWriter(output_video_full_path, fourcc, fps, (frame_width, frame_height), True)

    # Main part - actions performed on every frames
    frame_number = 0
    gui.progress["maximum"] = num_of_frames
    gui.progress["value"] = 0
    gui.pb_precentage_label.config(text='0%')
    gui.window.update()
    while cap_extracted.isOpened():
        try:
            gui.progress["value"] = frame_number
            cur_precentage = float('%.2f' % ((frame_number / num_of_frames) * 100))
            gui.pb_precentage_label.config(text=str(cur_precentage) + '%')
            gui.window.update()
            for i in range(0, 1):
                print(f'vm - frame num {frame_number}:{num_of_frames}')
                frame_number += 1
                ret_extracted, cur_frame_rgb = cap_extracted.read()
                if ret_extracted is False:
                   break

            # Binary image + Erosion as input for B/F
                ret_bin, cur_frame_bin = cap_bin.read()
            #if ret_bin is False:
            #    break

            # Read stabilized frame
                ret, cur_frame_orig = cap.read()
            #if ret_bin is False:
            #    break

            # Binary image comes in as RGB
                cur_frame_bin_grey = cv2.cvtColor(cur_frame_bin, cv2.COLOR_BGR2GRAY)
                ret_binary, cur_frame_binarized = cv2.threshold(cur_frame_bin_grey, 127, 255, cv2.THRESH_BINARY)

            # create eroded image - TODO: optimize erosion filter according to input binary image
            OmegaFG = cur_frame_binarized
            OmegaFG_Eroded = cv2.erode(OmegaFG, np.ones((5, 5), np.uint8), iterations=1)
            #OmegaFG_Scribble = cv2.dilate(OmegaFG_Eroded, np.ones((3, 3), np.uint8), iterations=10)
            OmegaBG = np.zeros_like(cur_frame_binarized)
            OmegaBG[cur_frame_binarized > 0] = 5
            OmegaBG[cur_frame_binarized == 0] = 255
            OmegaBG[cur_frame_binarized == 5] = 0

            # get scribble from BG for faster running time
            # create padding of 0 for erosion
            #OmegaBG_for_distance = OmegaBG
            #OmegaBG_for_distance[:, 0] = np.zeros_like(frame_height)  # first row
            #OmegaBG_for_distance[:, frame_width-1] = np.zeros_like(frame_height)  # last row
            #OmegaBG[0, :] = np.zeros_like(frame_width)
            #OmegaBG[frame_height-1, :] = np.zeros_like(frame_width)
            OmegaBG_Eroded = cv2.erode(OmegaBG, np.ones((15, 15), np.uint8), iterations=10)
            #OmegaBG_for_distance_eroded = cv2.erode(OmegaBG_for_distance, np.ones((19, 19), np.uint8), iterations=80)
            #BG_random_choice = np.random.choice(OmegaBG_nonzero_vec[1], size=70000)
            #OmegaBG_vec = np.zeros((1, frame_width*frame_height))
            #OmegaBG_vec[0,BG_random_choice] = 1
            #OmegaBG_sampled = np.reshape(OmegaBG_vec, (frame_height, frame_width))

            if(0):
                plt.figure()
                plt.subplot(131), plt.imshow(cur_frame_bin, 'gray'), plt.title('Binary image')
                plt.subplot(132), plt.imshow(OmegaFG_Eroded, 'gray'), plt.title('OmegaFG_Eroded')
                plt.subplot(133), plt.imshow(OmegaBG_for_distance_eroded, 'gray'), plt.title('OmegaBG_Eroded')
                plt.show(block=False)

            # calculate CDF using KDE
            CDF_given_F = get_CDF(cur_frame_orig, OmegaFG_Eroded)
            CDF_given_B = get_CDF(cur_frame_orig, OmegaBG_Eroded)

            # Create F/B likelyhood map
            Prob_map_F = np.zeros_like(cur_frame_rgb[:, :, 1])
            Prob_map_B = np.zeros_like(cur_frame_rgb[:, :, 1])
            Prob_map_F = np.divide(CDF_given_F, (CDF_given_F + CDF_given_B))
            Prob_map_B = np.divide(CDF_given_B, (CDF_given_F + CDF_given_B))

            if 0:
                plt.figure()
                plt.subplot(221), plt.imshow(cur_frame_rgb), plt.title('Original image')
                plt.subplot(222), plt.imshow(cur_frame_binarized), plt.title('Binary image')
                plt.subplot(223), plt.imshow(Prob_map_F, 'gray'), plt.title('Prob_map_F')
                plt.subplot(224), plt.imshow(Prob_map_B, 'gray'), plt.title('Prob_map_B')
                plt.show()

            Prob_map_F = np.uint8(Prob_map_F > 0.5)*255
            Prob_map_B = np.uint8(Prob_map_B > 0.5)*255

            # progress
            print('Calculated likelyhood')

            # Create normalized maps of gradient of likelyhood
            Prob_map_F_grad = cv2.Laplacian(Prob_map_F, cv2.CV_16S)
            Prob_map_B_grad = cv2.Laplacian(Prob_map_B, cv2.CV_16S)

            if 0:
                plt.figure()
                plt.subplot(221), plt.imshow(Prob_map_F), plt.title('Prob_map_F')
                plt.subplot(222), plt.imshow(Prob_map_B), plt.title('Prob_map_B')
                plt.subplot(223), plt.imshow(Prob_map_F_grad, 'gray'), plt.title('Prob_map_F_grad')
                plt.show()

            abs_Prob_map_F_grad = np.abs(Prob_map_F_grad)
            normalized_pr_pixel_Prob_map_F_grad = np.uint8(255*np.divide(abs_Prob_map_F_grad, np.max(abs_Prob_map_F_grad)))
            abs_Prob_map_B_grad = np.abs(Prob_map_B_grad)
            normalized_pr_pixel_Prob_map_B_grad = np.uint8(255*np.divide(abs_Prob_map_B_grad, np.max(abs_Prob_map_B_grad)))

            if 0:
                plt.figure()
                plt.subplot(131), plt.imshow(Prob_map_F), plt.title('Likelihood FG')
                plt.subplot(132), plt.imshow(Prob_map_B), plt.title('Likelihood BG')
                plt.subplot(133), plt.imshow(normalized_pr_pixel_Prob_map_B_grad, 'gray'), plt.title('normalized likelihood gradient')
                plt.show()

            ## calculate cost field for F/B
            cost_field_F = wdt.map_image_to_costs(normalized_pr_pixel_Prob_map_F_grad, OmegaFG_Eroded)
            #wdt.plot(cost_field_F)
            distance_transform_F = wdt.get_weighted_distance_transform(cost_field_F)
            #wdt.plot(distance_transform_F)

            #cost_field_B = wdt.map_image_to_costs(normalized_pr_pixel_Prob_map_B_grad, OmegaBG_Eroded)
            #wdt.plot(cost_field_B)
            #distance_transform_B = wdt.get_weighted_distance_transform(cost_field_B)
            #wdt.plot(distance_transform_B)

            # normalize distance maps
            #norm_dist_map_F = 255*np.divide(distance_transform_F, np.max(distance_transform_F))
            #norm_dist_map_B = 255*np.divide(distance_transform_B, np.max(distance_transform_B))

            # create BG distance map by inversing FG distmap - because of bug in WDT
            max_cost_F = np.max(cost_field_F)
            cost_field_B = max_cost_F * np.ones_like(cost_field_F) - cost_field_F
            distance_transform_B = wdt.get_weighted_distance_transform(cost_field_B)

            # progress
            print('Calculated distance maps')
            if 0:
                plt.figure()
                plt.subplot(221), plt.imshow(distance_transform_F, 'gray'), plt.title('distance_transform_F')
                plt.subplot(222), plt.imshow(distance_transform_B, 'gray'), plt.title('distance_transform_B')
                plt.subplot(223), plt.imshow(cost_field_F, 'gray'), plt.title('cost_field_F')
                plt.subplot(224), plt.imshow(cost_field_B, 'gray'), plt.title('cost_field_B')
                plt.show()

            # Create Trimap
            Trimap = get_Trimap(distance_transform_F, distance_transform_B, Trimap_radius_rho)
            if 0:
                plt.figure()
                plt.subplot(221), plt.imshow(distance_transform_F, 'gray'), plt.title('distance_transform_F')
                plt.subplot(222), plt.imshow(distance_transform_B, 'gray'), plt.title('distance_transform_B')
                plt.subplot(223), plt.imshow(Trimap, 'gray'), plt.title('Trimap')
                plt.show()
            print('Calculated Trimap')

            # Create Alpha maps
            Alpha_map = get_Alpha_map(distance_transform_F, distance_transform_B, Prob_map_F, Prob_map_B, Trimap, power_r)

            if 0:
                plt.figure()
                plt.subplot(221), plt.imshow(distance_transform_F, 'gray'), plt.title('distance_transform_F')
                plt.subplot(222), plt.imshow(distance_transform_B, 'gray'), plt.title('distance_transform_B')
                plt.subplot(223), plt.imshow(Trimap, 'gray'), plt.title('Trimap')
                plt.subplot(224), plt.imshow(Alpha_map, 'gray'), plt.title('Alpha Map')
                plt.show()

            print('Calculated Alpha map')

            # Create new frame with background
            ## resize background image
            new_BG_resized = np.resize(New_BG, (frame_height, frame_width, 3))

            #frame_matted = matt_frame(cur_frame_orig, new_BG_resized, Alpha_map, Trimap_radius_rho)
            frame_matted = simple_matt_frame(cur_frame_orig, new_BG_resized, Alpha_map)

            if (0):
                plt.figure()
                plt.subplot(321), plt.imshow(cur_frame_orig), plt.title('cur_frame_orig')
                plt.subplot(322), plt.imshow(frame_matted), plt.title('frame_matted')
                plt.subplot(323), plt.imshow(Trimap), plt.title('Trimap')
                plt.subplot(324), plt.imshow(Alpha_map, 'gray'), plt.title('Alpha_map')
                plt.subplot(325), plt.imshow(distance_transform_F, 'gray'), plt.title('distance_transform_F')
                plt.subplot(326), plt.imshow(distance_transform_B, 'gray'), plt.title('distance_transform_B')
                plt.savefig(os.curdir + "\\debug\\frame{}.jpg".format(frame_number))

            # Save frame in output video
            frame_out = frame_matted

            out.write(frame_out)
        except:
            print('exception')
            if frame_number > num_of_frames:
                break
            continue

    # Release video
    cap_extracted.release()
    cap_bin.release()
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()


