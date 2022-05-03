import os
from gui import ProjectGui

if not os.path.isdir(os.path.join('..', 'Output')):
    os.makedirs(os.path.join('..', 'Output'))

project_gui = ProjectGui()
project_gui.loop_gui()

