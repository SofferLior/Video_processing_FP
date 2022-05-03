import os

# TODO: These parameters should be changed using the gui

input_video_rel_path = os.path.join('..', 'Input')
output_video_rel_path = os.path.join('..', 'Output')

video_name = '20190602_150949.avi' # nadav wall one way
input_video_full_path = os.path.join(input_video_rel_path, video_name)
# stabilization parameters

# background subtractor
bs_time_window_size = 70
bs_subtraction_th = 70

# Video matting
New_Background_name = 'New_Background.jpg'
Trimap_radius_rho = 5
# TODO: define default flag to change to False when user selects another video/image
default_flag = True
power_r = 2