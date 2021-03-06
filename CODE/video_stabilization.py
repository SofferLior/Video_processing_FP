import numpy as np
import cv2
import matplotlib.pyplot as plt

debug_flag = False


def find_features_and_descriptor(grey_im):
    # goodFeaturesToTrack Params: maximum number of features, quality level, minimum possible Euclidean distance
    features = cv2.goodFeaturesToTrack(grey_im, 500, 0.01, 20)
    features=features.reshape(features.shape[0], 2)
    if debug_flag:
        plt.figure()
        plt.imshow(grey_im)
        plt.scatter(features.T[0],features.T[1])
        plt.show(False)
    sift = cv2.SIFT_create()
    kp = [cv2.KeyPoint(f[0], f[1], 1) for f in features]
    kp, des = sift.compute(grey_im, kp)
    return kp, des


def rearrange_points_according_to_matches(src_kp,des_kp,matches):
    src_pts = np.int32([src_kp[m.queryIdx].pt for m in matches])
    des_pts = np.int32([des_kp[m.trainIdx].pt for m in matches])
    return src_pts, des_pts


def convert_to_srt(trans):
    '''
    :param trans: 2x3 matrix
        [a11, a12, b1]
        [a21, a22, b2]
    :return: sRt matrix 3x3
        [s*cos(ang) , -s*sin(ang) , 0]
        [s*sin(ang) , s*cos(ang)  , 0]
        [t_X        , t_y         , 1]
    '''
    return np.hstack([trans.T,[[0],[0],[1]]])


def stabilize_video(input_path, video_data, output_video_path):
    cap = cv2.VideoCapture(input_path)
    out_stabilized = cv2.VideoWriter(output_video_path, video_data['fourcc'], video_data['fps'], (video_data['w'],video_data['h']))
    ret, prev = cap.read()
    out_stabilized.write(prev)
    if ret:
        prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_des = find_features_and_descriptor(prev_grey)
        H_cum = np.eye(3)
        for frame in range(video_data['frames_num']-1):
            # 1. read next frame
            ret, cur = cap.read()
            if not ret:
                break

            # 2. extract features and descriptor
            cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            cur_kp, cur_des = find_features_and_descriptor(cur_grey)

            # 3. Match descriptors
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = matcher.match(prev_des, cur_des)
            matches = sorted(matches, key= lambda x:x.distance)
            if debug_flag:
                matches_im = cv2.drawMatches(prev_grey,prev_kp,cur_grey,cur_kp,matches[:100],None, flags=2)
                plt.figure()
                plt.imshow(matches_im)
                plt.show()

            prev_pnt, cur_pnt = rearrange_points_according_to_matches(prev_kp, cur_kp, matches)
            trans, inliers = cv2.estimateAffine2D(cur_pnt,prev_pnt, method=cv2.RANSAC, ransacReprojThreshold=2)
            H_srt = convert_to_srt(trans)
            H_cum = np.dot(H_srt,H_cum)
            warp_cur = cv2.warpAffine(cur, H_cum[:, :2].T, (video_data['w'], video_data['h']))
            out_stabilized.write(warp_cur)

            prev_grey = cur_grey
            prev_kp, prev_des = cur_kp, cur_des

    out_stabilized.release()
    return out_stabilized
