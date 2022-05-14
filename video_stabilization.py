import numpy as np
import cv2
import matplotlib.pyplot as plt


def find_features_and_descriptor(grey_im):
    features = cv2.goodFeaturesToTrack(grey_im, 200, 0.01, 30, 3).reshape(200, 2)
    if 0:
        plt.figure()
        plt.imshow(grey_im)
        plt.scatter(features.T[0],features.T[1])
        plt.show(False)
    freak_extractor = cv2.xfeatures2d.FREAK_create()
    kp = [cv2.KeyPoint(f[1], f[0], 1) for f in features]
    kp, des = freak_extractor.compute(grey_im, kp)

    return kp, des


def orb_features_and_descriptor(grey_im):
    orb = cv2.ORB_create()  # TODO: remove this from the function - to be called once
    kp, des = orb.detectAndCompute(grey_im, None)
    return kp, des


def find_transformation(src_kp,des_kp,matches):
    good_matches = matches
    src_pts = np.int32([src_kp[m.queryIdx].pt for m in good_matches])
    des_pts = np.int32([des_kp[m.trainIdx].pt for m in good_matches])
    trans, inliers = cv2.estimateAffine2D(src_pts,des_pts)
    return trans


def stabilize_video(cap, video_data):
    out_stabilized = cv2.VideoWriter('stabilized.avi', video_data['fourcc'], video_data['fps'], (video_data['w'],video_data['h']))
    ret, prev = cap.read()
    out_stabilized.write(prev)
    if ret:
        prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_des = orb_features_and_descriptor(prev_grey)
        H_cum = numpy.eye(3)
        for frame in range(video_data['n_frames']-1):
            # 1. read next frame
            ret, cur = cap.read()
            if not ret:
                break

            # 2. extract features and descriptor
            cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            cur_kp, cur_des = orb_features_and_descriptor(cur_grey)

            # 3. Match descriptors
            matches = cv2.BFMatcher().match(prev_des,cur_des)
            matches = sorted(matches, key= lambda x:x.distance)
            #matches_im = cv2.drawMatches(prev_grey,prev_kp,cur_grey,cur_kp,matches,None, flags=2)

            trans = find_transformation(prev_kp,cur_kp,matches)
            # TODO: complete this according to matlab instructions
            H_cum = np.vstack([trans,[0,0,1]]) * H_cum
            # TODO: transform back to 2x3
            warp_cur_grey = cv2.warpAffine(cur_grey,H_cum[:2,:],(video_data['w'],video_data['h']))
            warp_cur = cv2.cvtColor(warp_cur_grey, cv2.COLOR_GRAY2BGR)
            out_stabilized.write(warp_cur)

            prev_grey = cur_grey
            prev_kp, prev_des = orb_features_and_descriptor(prev_grey)

    print('sadasd')
    return cap