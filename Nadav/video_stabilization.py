import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk

# def stabilize_video(input_video_name, output_video_name, max_corners, quality_level, min_distance):
#     cap = cv2.VideoCapture(input_video_name)
#
#     if cap.isOpened() is False:
#         print('Error openning video stream or file')
#
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # Find OpenCV version
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#     if int(major_ver) < 3:
#         fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#     else:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#     is_color = True
#     out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height), is_color)
#
#     frame_number = 0
#     while (cap.isOpened()):
#         if frame_number == 0:
#             ret, cur_frame_rgb = cap.read()
#             if ret is False:
#                 break
#             cur_frame_gray = cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2GRAY)
#             cur_corners = cv2.goodFeaturesToTrack(cur_frame_gray, max_corners, quality_level, min_distance)
#
#             if 0:
#                 plt.figure()
#                 plt.imshow( cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2RGB))
#                 plt.scatter(cur_corners[:,:,0], cur_corners[:,:,1], color='red', marker='x')
#
#             frame_number += 1
#             out.write(cur_frame_rgb)
#             continue
#         # else
#         previous_frame_gray = cur_frame_gray
#         previous_corners = cur_corners
#
#         ret, cur_frame_rgb = cap.read()
#         if ret is False:
#             break
#         cur_frame_gray = cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2GRAY)
#         cur_corners = cv2.goodFeaturesToTrack(cur_frame_gray, max_corners, quality_level, min_distance)
#
#         cur_pts_OF, status, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, cur_frame_gray, previous_corners, None)
#         if cur_pts_OF.shape != previous_corners.shape:
#             pass
#             #TODO: handle?
#         # Filter only valid points
#         index_of_valid_corners = np.where(status == 1)[0]
#         previous_corners = previous_corners[index_of_valid_corners]
#         cur_pts_OF = cur_pts_OF[index_of_valid_corners]
#
#         if 0:
#             plt.figure()
#             plt.imshow(cv2.cvtColor(cur_frame_rgb, cv2.COLOR_BGR2RGB))
#             plt.scatter(previous_corners[:, :, 0], previous_corners[:, :, 1], color='yellow', marker='o', label='previous_corners')
#             plt.scatter(cur_pts_OF[:, :, 0], cur_pts_OF[:, :, 1], color='blue', marker=4, label='cur_pts_OF')
#             plt.scatter(cur_corners[:, :, 0], cur_corners[:, :, 1], color='red', marker='x', label='cur_corners')
#             plt.legend()
#
#         out.write(cur_frame_rgb)
#         frame_number += 1
#         if frame_number == num_of_frames:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     return True

# The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS = 50

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def video_stabilization_using_point_feature_matching(gui, input_video_path, output_video_path):
    output_video_full_path = os.path.join(output_video_path, 'stabilized.avi')

    # using the link in the 'project tips' presentation: https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
    # Read input video
    cap = cv2.VideoCapture(input_video_path)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec for output video
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    # Set up output video
    out = cv2.VideoWriter(output_video_full_path, fourcc, fps, (w, h), True)
    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    gui.progress["maximum"] = 2*n_frames
    gui.progress["value"] = 0
    gui.pb_precentage_label.config(text='0%')
    gui.window.update()

    for i in range(n_frames - 2):
        gui.progress["value"] = i
        cur_precentage = float('%.2f' % ((i/(2*n_frames))*100))
        gui.pb_precentage_label.config(text=str(cur_precentage)+'%')
        gui.window.update()
        print(f'vs - read - frame num {i}:{n_frames}')
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        if 0:
            plt.figure()
            plt.imshow(prev_gray, cmap='gray')
            plt.scatter(prev_pts[:,:,0], prev_pts[:,:,1], marker='+', color='red')

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)  # will only work with OpenCV-3 or less

        # Extract traslation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        gui.progress["value"] = n_frames + i
        cur_precentage = 50 + float('%.2f' % ((i/(2*n_frames))*100))
        gui.pb_precentage_label.config(text=str(cur_precentage)+'%')
        gui.window.update()
        print(f'vs - write - frame num {i}:{n_frames}')
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        # frame_out = cv2.hconcat([frame, frame_stabilized])
        frame_out = frame_stabilized

        # # If the image is too big, resize it.
        # if (frame_out.shape[1] > 1920):
        #     frame_out = cv2.resize(frame_out, (round(frame_out.shape[1] / 2), round(frame_out.shape[0] / 2)))

        # cv2.imshow("Before and After", frame_out)
        # cv2.waitKey(10)
        out.write(frame_out)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()

