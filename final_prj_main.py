import argparse
import os
import cv2
import time
import json
from collections import OrderedDict

from background_subtraction import background_subtraction
from obj_tracking import track_obj
from video_matting import video_matting
from video_stabilization import stabilize_video

#  TODO: add IDs

def main(args):
    timing_path = f'Output/timing.json'
    timing = OrderedDict()

    input_video_path = os.path.join(args.input_folder_path, 'INPUT.avi')
    new_background = os.path.join(args.input_folder_path, 'background.jpg')

    # load video and its data
    cap, video_data = load_video(input_video_path)

    # send to video stabilization
    start_time = time.time()
    stabilized_video = stabilize_video(cap, video_data)
    end_time = time.time()
    timing["time_to_stabilize"] = end_time - start_time

    # subtract background
    obj_video = background_subtraction(stabilized_video, video_data)

    # video matting
    new_background_video = video_matting(obj_video, new_background, video_data)

    # object tracking
    video_w_tracking = track_obj(new_background_video, video_data)

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
    video_data['n_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    parser.add_argument('--output_folder_path', help='path to the output folder to save processed video', default='Outputs')
    args = parser.parse_args()
    main(args)
