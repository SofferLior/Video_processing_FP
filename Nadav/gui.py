import os
import tkinter as tk
import cv2
from tkinter import ttk, Toplevel, messagebox
from tkinter import filedialog as fd
import parameters as par
from video_stabilization import video_stabilization_using_point_feature_matching
from backgroud_subtraction import background_subtraction
from object_tracking import object_tracking
from video_matting import video_matting


class ProjectGui():
    def __init__(self):
        # Gui:
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = Toplevel(self.root)
        self.window.title("Finale project")

        tk.Label(self.window, text="Final project").grid(row=0, column=4)

        # and progressbar
        # progressbar
        self.pb_label = tk.Label(self.window, text='Nothing is in progress')
        self.pb_label.grid(row=7, column=1)
        self.progress = ttk.Progressbar(self.window, orient='horizontal', length=100, mode='determinate')
        self.progress.grid(row=7, column=3)
        self.pb_precentage_label = tk.Label(self.window, text="")
        self.pb_precentage_label.grid(row=7, column=4)

        # now, create entries for parameters
        tk.Label(self.window, text="parameters:").grid(row=1, column=6)
        tk.Label(self.window, text="- video to run:").grid(row=2, column=6)
        self.tk_video_fullpath = tk.Label(self.window, text=par.input_video_full_path)
        self.tk_video_fullpath.grid(row=2, column=7)
        self.btn_choose_file = tk.Button(self.window, text="...", command=self.choose_file_btn_command)
        self.btn_choose_file.grid(row=2, column=9)


        tk.Label(self.window, text="background subtraction parameters:").grid(row=3, column=6)
        tk.Label(self.window, text="- Time window size").grid(row=4, column=6)
        self.entry_bs_t_window_size = tk.Entry(self.window)
        self.entry_bs_t_window_size.grid(row=4, column=7)
        self.entry_bs_t_window_size.insert(0, str(par.bs_time_window_size))
        tk.Label(self.window, text="- Subtraction TH").grid(row=5, column=6)
        self.entry_bs_th = tk.Entry(self.window, text='70')
        self.entry_bs_th.grid(row=5, column=7)
        self.entry_bs_th.insert(0, str(par.bs_subtraction_th))

        tk.Label(self.window, text="video matting parameters:").grid(row=6, column=6)
        tk.Label(self.window, text="- Trimap_radius_rho").grid(row=7, column=6)
        self.entry_vm_trimap_rasius_rho = tk.Entry(self.window)
        self.entry_vm_trimap_rasius_rho.grid(row=7, column=7)
        self.entry_vm_trimap_rasius_rho.insert(0, str(par.Trimap_radius_rho))
        tk.Label(self.window, text="- power r").grid(row=8, column=6)
        self.entry_vm_power_r = tk.Entry(self.window)
        self.entry_vm_power_r.grid(row=8, column=7)
        self.entry_vm_power_r.insert(0, str(par.power_r))

        # now, create some buttons
        tk.Label(self.window, text="Which step do you want to *start* with?").grid(row=1, column=1)

        self.btn_vs = tk.Button(self.window, text="Video stabilization", fg="red", command=self.vs_btn_command)
        self.btn_vs.grid(row=2, column=1)

        self.btn_bs = tk.Button(self.window, text="Background subtraction", fg="green", command=self.bs_btn_command)
        self.btn_bs.grid(row=3, column=1)

        self.btn_mt = tk.Button(self.window, text="Video Matting", fg="purple", command=self.vm_btn_command)
        self.btn_mt.grid(row=4, column=1)

        self.btn_tr = tk.Button(self.window, text="Tracking", fg="orange", command=self.tr_btn_command)
        self.btn_tr.grid(row=5, column=1)

    def loop_gui(self):
        self.window.mainloop()

    def choose_file_btn_command(self):
        filename = fd.askopenfilename()
        if filename is not '':
            self.tk_video_fullpath.config(text=filename)
            self.window.update()
        return

    def vs_btn_command(self):
        window_size = int(self.entry_bs_t_window_size.get())
        th = int(self.entry_bs_th.get())
        Trimap_radius_rho = int(self.entry_vm_trimap_rasius_rho.get())
        power_r = int(self.entry_vm_power_r.get())

        print('stabilize video starts')
        self.pb_label.config(text='Video stabilization in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        st_input_video_full_path = self.tk_video_fullpath['text']
        st_output_video_path = par.output_video_rel_path
        video_stabilization_using_point_feature_matching(self, st_input_video_full_path, st_output_video_path)
        self.video_viewer(os.path.join(st_output_video_path, 'stabilized.avi'), 'stabilized video')
        print('stabilize video finish')

        # background subtraction
        print('background subtraction starts')
        self.pb_label.config(text='Background subtraction in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        bs_input_video_full_path = os.path.join(par.output_video_rel_path, 'stabilized.avi')
        bs_output_video_path = par.output_video_rel_path
        background_subtraction(self, bs_input_video_full_path, bs_output_video_path, window_size, th)
        self.video_viewer(os.path.join(bs_output_video_path, 'extracted.avi'), 'extracted video')
        print('background subtraction finished')

        # video matting
        print('video matting starts')
        self.pb_label.config(text='Video matting in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        vm_input_video_path = os.path.join(par.output_video_rel_path, 'stabilized.avi')
        vm_input_video_extracted_path = os.path.join(par.output_video_rel_path, 'extracted.avi')
        vm_input_video_binary_path = os.path.join(par.output_video_rel_path, 'binary.avi')
        vm_output_video_path = par.output_video_rel_path
        vm_New_Background_path = os.path.join(par.input_video_rel_path, par.New_Background_name)
        video_matting(self, vm_input_video_path, vm_input_video_extracted_path, vm_output_video_path, vm_input_video_binary_path, vm_New_Background_path, Trimap_radius_rho, power_r)
        self.video_viewer(os.path.join(bs_output_video_path, 'matted.avi'), 'matted video')

        # tracking
        self.pb_label.config(text='Tracking in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        tr_input_video_full_path = os.path.join(par.output_video_rel_path, 'matted.avi')
        tr_output_video_path = par.output_video_rel_path
        object_tracking(self, tr_input_video_full_path, tr_output_video_path)
        self.video_viewer(os.path.join(tr_output_video_path, 'OUTPUT.avi'), 'tracked video')
        print('tracking finished')
        return

    def bs_btn_command(self):
        window_size = int(self.entry_bs_t_window_size.get())
        th = int(self.entry_bs_th.get())
        Trimap_radius_rho = int(self.entry_vm_trimap_rasius_rho.get())
        power_r = int(self.entry_vm_power_r.get())

        # background subtraction
        print('background subtraction starts')
        self.pb_label.config(text='Background subtraction in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        bs_input_video_full_path = self.tk_video_fullpath['text']
        if not 'stabilized' in bs_input_video_full_path:
            print('file name should be "stabilized.avi"')
            messagebox.showerror('file name', 'file name should be "stabilized.avi')
            return
        bs_output_video_path = par.output_video_rel_path
        background_subtraction(self, bs_input_video_full_path, bs_output_video_path, window_size, th)
        self.video_viewer(os.path.join(bs_output_video_path, 'extracted.avi'), 'extracted video')
        print('background subtraction finished')

        # video matting
        print('video matting starts')
        self.pb_label.config(text='Video matting in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        vm_input_video_path = os.path.join(par.output_video_rel_path, 'stabilized.avi')
        vm_input_video_extracted_path = os.path.join(par.output_video_rel_path, 'extracted.avi')
        vm_input_video_binary_path = os.path.join(par.output_video_rel_path, 'binary.avi')
        vm_output_video_path = par.output_video_rel_path
        vm_New_Background_path = os.path.join(par.input_video_rel_path, par.New_Background_name)
        video_matting(self, vm_input_video_path, vm_input_video_extracted_path, vm_output_video_path,
                      vm_input_video_binary_path, vm_New_Background_path, par.Trimap_radius_rho, par.power_r)
        self.video_viewer(os.path.join(bs_output_video_path, 'matted.avi'), 'matted video')

        # tracking
        self.pb_label.config(text='Tracking in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        tr_input_video_full_path = os.path.join(par.output_video_rel_path, 'matted.avi')
        tr_output_video_path = par.output_video_rel_path
        object_tracking(self, tr_input_video_full_path, tr_output_video_path)
        self.video_viewer(os.path.join(tr_output_video_path, 'OUTPUT.avi'), 'tracked video')
        print('tracking finished')

        return

    def vm_btn_command(self):
        messagebox.showinfo("Title", "Make sure you have stabilized, extracted and binary videos from the same original video \n on output folder")
        Trimap_radius_rho = int(self.entry_vm_trimap_rasius_rho.get())
        power_r = int(self.entry_vm_power_r.get())

        # video matting
        print('video matting starts')
        self.pb_label.config(text='Video matting in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()

        vm_input_video_full_path = self.tk_video_fullpath['text']
        if not 'extracted' in vm_input_video_full_path:
            print('file name should be "extracted.avi"')
            messagebox.showerror('file name', 'file name should be "extracted.avi')

            return

        vm_input_video_extracted_path = vm_input_video_full_path
        vm_input_video_path = os.path.join(par.output_video_rel_path, 'stabilized.avi')
        vm_input_video_binary_path = os.path.join(par.output_video_rel_path, 'binary.avi')
        vm_output_video_path = par.output_video_rel_path
        vm_New_Background_path = os.path.join(par.input_video_rel_path, par.New_Background_name)
        video_matting(self, vm_input_video_path, vm_input_video_extracted_path, vm_output_video_path,
                      vm_input_video_binary_path, vm_New_Background_path, Trimap_radius_rho, power_r)
        self.video_viewer(os.path.join(par.output_video_rel_path, 'matted.avi'), 'matted video')

        # tracking
        self.pb_label.config(text='Tracking in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        tr_input_video_full_path = os.path.join(par.output_video_rel_path, 'matted.avi')
        tr_output_video_path = par.output_video_rel_path
        object_tracking(self, tr_input_video_full_path, tr_output_video_path)
        self.video_viewer(os.path.join(tr_output_video_path, 'OUTPUT.avi'), 'tracked video')
        print('tracking finished')

        return

    def tr_btn_command(self):

        print('tracking starts')
        self.pb_label.config(text='Tracking in progress')
        self.progress["value"] = 0
        self.pb_precentage_label.config(text='0%')
        self.window.update()
        tr_input_video_full_path = self.tk_video_fullpath['text']
        if not 'matted' in tr_input_video_full_path:
            print('file name should be "matted.avi"')
            messagebox.showerror('file name', 'file name should be "matted.avi')
            return
        tr_output_video_path = par.output_video_rel_path
        object_tracking(self, tr_input_video_full_path, tr_output_video_path)
        self.video_viewer(os.path.join(tr_output_video_path, 'OUTPUT.avi'), 'tracked video')
        print('tracking finished')
        return

    def video_viewer(self, video_full_path, video_name):
        video_name = video_name + '(press q to quit)'
        cap = cv2.VideoCapture(video_full_path)

        # Get frame count
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get width and height of video stream
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get frames per second (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_number = 0
        while(cap.isOpened()):
            ret, cur_frame_rgb = cap.read()
            if ret is False:
                break
            frame_number += 1
            cv2.imshow(video_name, cur_frame_rgb)
            if cv2.waitKey(int(fps)) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(video_name)
        return