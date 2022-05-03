import os
import cv2
import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
from gui_tool_pick_s_initial import TrCanvasGui
from tkinter import messagebox

def crop_image(I,x_start,x_end,y_start,y_end):
    height, width, channels = I.shape

    # Change indices to INT
    x_start = int(x_start)
    x_end = int(x_end)
    y_start = int(y_start)
    y_end = int(y_end)

    if channels>1:
        type = 'RGB'
    else:
        type = 'Grey'

    # check indices
    if x_end > (width-1):
        x_end = width-1
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0
    if y_end > (height-1):
        y_end = height-1

    # crop by image type
    if type=='RGB':
        I_crop = I[y_start:y_end, x_start:x_end, :]
    else: # gray
        I_crop = I[y_start:y_end, x_start:x_end]

    return(I_crop)


def get_norm_vector(v):
    sum = np.sum(v)
    if sum == 0:
        return v
    return v / sum


def compNormHist(I, S):
    #  INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
    #  OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1 VECTOR...
    #       ...NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1)
    height, width, channels = I.shape
    cur_ROI = crop_image(I, S[0] - S[2] - 1, S[0] + S[2],
                         S[1] - S[3] - 1, S[1] + S[3])

    if 0:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(I)
        plt.title('original image')
        plt.subplot(1,2,2)
        plt.imshow(cur_ROI)
        plt.title('cropped_roi')
        plt.show(block=False)

    ROI_height, ROI_width, channels = cur_ROI.shape
    # Initialize histogram
    hist_matrix = np.zeros([16, 16, 16], dtype=int)

    # Converting to 4 bit & counting
    cur_ROI_4b = np.zeros([ROI_height, ROI_width, channels], dtype=int)
    for row in range(0, ROI_height):
        for col in range(0, ROI_width):
            for color in range(channels):
                cur_ROI_4b[row, col, color] = int(cur_ROI[row, col, color] * 15 / 255)
            hist_matrix[cur_ROI_4b[row, col, 0], cur_ROI_4b[row, col, 1], cur_ROI_4b[row, col, 2]] = hist_matrix[
                                                                                                         cur_ROI_4b[
                                                                                                             row, col, 0],
                                                                                                         cur_ROI_4b[
                                                                                                             row, col, 1],
                                                                                                         cur_ROI_4b[
                                                                                                             row, col, 2]] + 1
    hist_vector = np.reshape(hist_matrix, 4096)

    # normalize histogram vector
    norm_hist_vector = get_norm_vector(hist_vector)

    return norm_hist_vector


def compBatDist(p, q):
    sum_p_times_q = 0
    if len(p) == len(q):
        for i in range(len(p)):
            sum_p_times_q = sum_p_times_q + np.sqrt(p[i]*q[i])
        distance = np.exp(20*sum_p_times_q)
        return distance
    else:
        print('Arrays should be the same length for Bhatt dist calc')
        return


def predictParticles(S_next_tag):
    N = np.shape(S_next_tag)[1]
    noise = numpy.random.normal(size=np.shape(S_next_tag))
    noiseWeights = np.array([[0.1], [0.1], [0], [0], [2], [1]])
    additiveNoise = np.multiply(np.matlib.repmat(noiseWeights, 1, N), noise)
    S_next = S_next_tag + additiveNoise

    # update the prediction of x and y using the velocity
    S_next[0:2, :] = S_next[0:2, :] + S_next[4:6, :]

    return S_next


def sampleParticles(S_prev, C):
    S_next_tag = np.zeros_like(S_prev)
    size, N = S_prev.shape
    for n in range(N):
        # Sample r from[0,1]
        r = np.random.random_sample()
        # Find lowest j with C[j]>=r
        for j in range(N):
            if C[j] >= r:
                S_next_tag[:, n] = S_prev[:, j]
                break

    return S_next_tag

def compNormWeightsAndPerdictCDF(I, N, S, q):
    # initialize weights
    W = np.zeros(N)
    # calculate weights
    for i in range(N):
        p = compNormHist(I, S[:, i])
        W[i] = compBatDist(p, q)
    W_norm = get_norm_vector(W)

    # initialize C
    C = np.zeros(N)
    # calculate CDF
    for i in range(N - 1):
        if i == 0:
            C[i] = W_norm[i]
        else:
            C[i] = C[i - 1] + W_norm[i]
    C[N - 1] = 1
    return W_norm, C

def showParticles(I, S, W):
    I = I.copy()
    # plt.figure()
    # plt.title(f'{ID}-Frame number = {i}')

    # average
    average_S = np.array(np.matrix(S) @ np.transpose(np.matrix(W)))

    cv2.rectangle(I, (average_S[0] - average_S[2], average_S[1] - average_S[3]), (average_S[0] + average_S[2], average_S[1] + average_S[3]), (0, 255, 0), 1)

    # max
    max_w_index = np.argsort(W)[-1]
    S_max_w = S[:, max_w_index]
    cv2.rectangle(I, (int(S_max_w[0] - S_max_w[2]), int(S_max_w[1] - S_max_w[3])), (int(S_max_w[0] + S_max_w[2]), int(S_max_w[1] + S_max_w[3])), (0, 0, 255), 1)

    # plt.imshow(I)
    # plt.show(block=False)
    # cv2.imwrite(f'{ID}-{str(i)}.png', I)
    return I

def create_s_initial_with_gui(gui, image):
    MsgBox = messagebox.askquestion('Q', 'Do you want to choose the object \n to detect manually?', icon='warning')
    if MsgBox == 'yes':
        choose_s_initial_tool = TrCanvasGui(gui, image)
        x_coordinates = choose_s_initial_tool.tr_s_initial_x
        y_coordinates = choose_s_initial_tool.tr_s_initial_y

        s_initial = np.array([[np.floor(np.median(x_coordinates))],  # x center
                              [np.floor(np.median(y_coordinates))],  # y center
                              [np.floor(np.abs(0.5 * (x_coordinates[0] - x_coordinates[1])))],  # half width
                              [np.floor(np.abs(0.5 * (y_coordinates[0] - y_coordinates[1])))],  # half height
                              [0],  # velocity x
                              [0]])  # velocity y
        print('manually')
    else:
        s_initial = np.array([[1798],  # x center
                              [697],  # y center
                              [44],  # half width
                              [176],  # half height
                              [0],  # velocity x
                              [0]])  # velocity y
        print('default')


    return s_initial

def object_tracking(gui, input_video_full_path, output_video_path):
    # TODO: add selection for s_initial
    output_tracked_full_path = os.path.join(output_video_path, 'OUTPUT.avi')
    N = 100  # SET NUMBER OF PARTICLES

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
    out = cv2.VideoWriter(output_tracked_full_path, fourcc, fps, (frame_width, frame_height), True)

    frame_number = 0
    # first image:
    print(f'tr - frame num {frame_number}:{num_of_frames}')
    frame_number += 1
    ret, cur_frame_rgb = cap.read()
    if ret is False:
        print('could not open video')
        return

    s_initial = create_s_initial_with_gui(gui, cur_frame_rgb.copy())

    # CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
    S = predictParticles(np.matlib.repmat(s_initial, 1, N))

    # LOAD FIRST IMAGE
    I = cur_frame_rgb

    # COMPUTE NORMALIZED HISTOGRAM
    q = compNormHist(I, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W, C = compNormWeightsAndPerdictCDF(I, N, S, q)

    if 0:
        plt.figure()
        plt.plot(C)
        plt.show(block=False)

    cur_output_frame = showParticles(I, S, W)

    if 0:
        plt.figure()
        plt.imshow(cur_output_frame)
        plt.show(block=False)

    out.write(cur_output_frame)

    # MAIN TRACKING LOOP
    gui.progress["maximum"] = num_of_frames
    gui.pb_precentage_label.config(text='0%')
    gui.progress["value"] = 0
    gui.window.update()
    while (cap.isOpened()):
        gui.progress["value"] = frame_number
        cur_precentage = float('%.2f' % ((frame_number/num_of_frames)*100))
        gui.pb_precentage_label.config(text=str(cur_precentage)+'%')
        gui.window.update()
        print(f'tr - frame num {frame_number}:{num_of_frames}')
        frame_number += 1
        ret, cur_frame_rgb = cap.read()
        if ret is False:
            return
        S_prev = S

        # LOAD NEW IMAGE FRAME
        I = cur_frame_rgb

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sampleParticles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predictParticles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        W, C = compNormWeightsAndPerdictCDF(I, N, S, q)

        # CREATE DETECTOR PLOTS
        cur_output_frame = showParticles(I, S, W)

        if 0:
            plt.figure()
            plt.imshow(cur_output_frame)
            plt.show(block=False)

        out.write(cur_output_frame)

