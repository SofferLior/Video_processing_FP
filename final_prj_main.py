import argparse
import os
import cv2
import time
import json
from collections import OrderedDict

from background_subtraction import background_subtraction, background_subtraction_first_frames
from obj_tracking import track_obj
from video_matting import video_matting
from video_stabilization import stabilize_video

# TODO: add IDs
# TODO: change the functions not to have video outputs


def main(args):
    timing_path = f'Output/timing.json'
    timing = OrderedDict()

    input_video_path = os.path.join(args.input_folder_path, 'INPUT.avi')
    new_background = os.path.join(args.input_folder_path, 'background.jpg')
    stabilized_video_path = os.path.join('Output', 'stabilized.avi')

    # load video and its data
    cap, video_data = load_video(input_video_path)

    # send to video stabilization
    start_time = time.time()
    stabilize_video(cap, video_data, stabilized_video_path)
    end_stabilized_time = time.time()
    timing["time_to_stabilize"] = end_stabilized_time - start_time

    # subtract background
    stabilized_video_capture, video_data = load_video(stabilized_video_path)
    stabilized_video_capture_for_first_frames, video_data_for_first_frames = load_video(stabilized_video_path)
    first_binary_frames, first_fg_frames = background_subtraction_first_frames(stabilized_video_capture_for_first_frames, video_data_for_first_frames, args.output_folder_path)
    obj_video = background_subtraction(stabilized_video_capture, video_data, args.output_folder_path, first_binary_frames, first_fg_frames)
    end_bs_time = time.time()
    timing["time_to_binary"] = end_bs_time - end_stabilized_time

    # video matting
    video_matting(obj_video, new_background, video_data)

    # object tracking
    # TODO: change back to take the new_background_video
    binary_video_path = os.path.join('Output', 'binary_foreground.avi')
    video_w_tracking = track_obj(stabilized_video_path, binary_video_path,  video_data)

    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=4)

    #  TODO: debug
    try:
        test_json(timing_path)
    except:
        print('PROBLEM in the json')
    print('Bye')


def load_video(input_video_path):
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

    return cap, video_data


#  TODO: delete this function
def test_json(timing_path):
    d = {
        "time_to_stabilize": 1,
        "time_to_binary": 2,
        "time_to_alpha": 3,
        "time_to_matted": 4,
        "time_to_output": 5,
    }
    d_test = json.load(open(timing_path,'r'))
    for k in d:
        if k not in d_test:
            assert False, f"Your JSON does not include {k}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder_path', help='path to the input folder', default='Inputs')
    parser.add_argument('--output_folder_path', help='path to the output folder to save processed video', default='Output')
    args = parser.parse_args()
    main(args)
