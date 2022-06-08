import cv2
from collections import OrderedDict


def draw_rectangle(frame, x, y, w, h):
    color = (0, 128, 0)  # green
    thickness = 3
    return cv2.rectangle(frame,(x, y), (x+w, y+h), color,thickness)


def track_obj(input_path, binary_path, output_path, video_data):
    cap = cv2.VideoCapture(input_path)
    binary_cap = cv2.VideoCapture(binary_path)
    out_tracked = cv2.VideoWriter(output_path, video_data['fourcc'], video_data['fps'], (video_data['w'], video_data['h']))
    tracking = OrderedDict()

    curr_frame = 0
    while (cap.isOpened()):
        ret, cur_frame_rgb = cap.read()
        binary_ret, cur_binary_frame = binary_cap.read()
        if (ret or binary_ret) is False:
            break
        curr_frame += 1
        grey_binary = cv2.cvtColor(cur_binary_frame, cv2.COLOR_RGB2GRAY)
        x, y, w, h = cv2.boundingRect(grey_binary)
        tracking[curr_frame] = [int(x + w/2), int(y + h/2), int(w/2), int(h/2)]
        out_tracked.write(draw_rectangle(cur_frame_rgb, x, y, w, h))

    out_tracked.release()

    return tracking
