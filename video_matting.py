import cv2



def video_matting(fg_video, fg_video_data, new_background, new_bg_video_data, matted_video_path):

    out_extracted_fg = cv2.VideoWriter(matted_video_path, fg_video_data['fourcc'], fg_video_data['fps'], (fg_video_data['w'], fg_video_data['h']), True)

    curr_frame = 0

    while (fg_video.isOpened()):

        ret, cur_frame_fg = fg_video.read()
        if ret is False:
            break
        ret, cur_frame_new_bg = new_background.read()
        if ret is False:
            break



    return fg_video