import argparse
import os
import cv2
import time
import json
from collections import OrderedDict

from background_subtraction import background_subtraction, background_subtraction_first_frames
from obj_tracking import track_obj
from video_matting import video_matting, create_alpha
from video_stabilization import stabilize_video


ID1 = "203135058"
ID2 = "203764170"


def main(args):
    timing_path = os.path.join(args.output_folder_path, "timing_{0}_{1}.json".format(ID1, ID2))
    tracking_path = os.path.join(args.output_folder_path, "tracking_{0}_{1}.json".format(ID1, ID2))
    timing = OrderedDict()

    input_video_path = os.path.join(args.input_folder_path, 'INPUT.avi')
    new_background_image_path = os.path.join(args.input_folder_path, 'background.jpg')
    stabilized_video_path = os.path.join(args.output_folder_path, "stabilized_{0}_{1}.avi".format(ID1, ID2))
    extracted_fg_path = os.path.join(args.output_folder_path, "extracted_{0}_{1}.avi".format(ID1, ID2))
    binary_path = os.path.join(args.output_folder_path, "binary_{0}_{1}.avi".format(ID1, ID2))
    matted_video_path = os.path.join(args.output_folder_path, "matted_{0}_{1}.avi".format(ID1, ID2))
    alpha_video_path = os.path.join(args.output_folder_path, "alpha_{0}_{1}.avi".format(ID1, ID2))
    output_video_path = os.path.join(args.output_folder_path, "OUTPUT_{0}_{1}.avi".format(ID1, ID2))

    # get input video data
    video_data = get_video_params(input_video_path)

    # send to video stabilization
    start_time = time.time()
    stabilize_video(input_video_path, video_data, stabilized_video_path)
    end_stabilized_time = time.time()
    timing["time_to_stabilize"] = int(end_stabilized_time - start_time)
    print('Complete Stabilization')

    # subtract background
    first_binary_frames, first_fg_frames = background_subtraction_first_frames(stabilized_video_path)
    background_subtraction(stabilized_video_path, video_data, extracted_fg_path, binary_path, first_binary_frames, first_fg_frames)
    end_bs_time = time.time()
    timing["time_to_binary"] = int(end_bs_time - end_stabilized_time)
    print('Complete Background Subtraction')

    # video matting
    create_alpha(stabilized_video_path, binary_path, alpha_video_path, video_data)
    end_alpha_time = time.time()
    timing["time_to_alpha"] = int(end_alpha_time - end_bs_time)
    print('Complete Alpha')

    video_matting(stabilized_video_path, alpha_video_path, new_background_image_path, matted_video_path, video_data)
    end_matting_time = time.time()
    timing["time_to_matting"] = int(end_matting_time - end_alpha_time)
    print('Complete Matting')

    # object tracking
    tracking = track_obj(matted_video_path, binary_path, output_video_path, video_data)
    end_tracking_time = time.time()
    timing["time_to_output"] = int(end_tracking_time - end_matting_time)
    print('Complete Tracking')

    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=4)

    with open(tracking_path, 'w') as f:
        json.dump(tracking, f, indent=4)

    print('Complete Final Project')


def get_video_params(input_video_path):
    # Read input video
    cap = cv2.VideoCapture(input_video_path)

    video_data = dict()
    # Get frame count
    video_data['frames_num'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    video_data['w'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_data['h'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    video_data['fps'] = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec for output video
    video_data['fourcc'] = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    return video_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder_path', help='path to the input folder', default='Inputs')
    parser.add_argument('--output_folder_path', help='path to the output folder to save processed video', default='Output')
    args = parser.parse_args()
    main(args)
