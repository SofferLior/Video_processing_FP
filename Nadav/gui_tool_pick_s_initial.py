import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, Toplevel
from PIL import Image, ImageTk, ImageOps


class TrCanvasGui():
    def __init__(self, root_gui, np_image):
        pil_image = Image.fromarray(np_image)
        self.tr_canvas_window = Toplevel(root_gui.root)
        w, h = self.tr_canvas_window.winfo_screenwidth(), self.tr_canvas_window.winfo_screenheight()
        self.tr_canvas_window.geometry("%dx%d+0+0" % (w, h))

        self.tr_canvas_window.title("Choose s initial")

        self.tr_canvas = tk.Canvas(self.tr_canvas_window)
        photo = ImageTk.PhotoImage(pil_image, master=self.tr_canvas)
        self._canvas_image = self.tr_canvas.create_image(0, 0, image=photo, anchor=tk.NW)


        self.tr_canvas.bind("<Button-1>", self._left_click_record)
        self.tr_s_initial_x = []
        self.tr_s_initial_y = []
        self.counter_of_clicks = 0
        self.tr_canvas.pack(fill="both",  expand=1)

        messagebox.showinfo("Title", "Choose boundig box - first choose upper left corner and then \n choose lower left coener. \n If it's necessary enlarge the window size")

        self.tr_canvas_window.mainloop()
        print('out of pick points gui')

    def _left_click_record(self, event=None):
        self.tr_s_initial_x.append(event.x)
        self.tr_s_initial_y.append(event.y)
        self.counter_of_clicks += 1
        if self.counter_of_clicks >= 2:
            self.tr_canvas.delete(self._canvas_image)
            self.tr_canvas_window.withdraw()
            self.tr_canvas_window.quit()
