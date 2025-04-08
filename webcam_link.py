"""
NAME: webcam_link.py
DESCRIPTION: use openCV to get camera access
PROGRAMMER: Caidan Gray
CREATION DATE: 4/3/2025
LAST EDITED: 4/7/2025   (please update each time the script is changed)
"""
import os

import cv2


class Webcam_Link():
    def __init__(self, webcam: int):
        """initialize the webcam link by creating variables and a named window"""
        super(Webcam_Link, self).__init__()
        # create a window
        cv2.namedWindow("Sign Language Recognition")
        self.vc = cv2.VideoCapture(webcam)



    def stop_webcam(self):
        """stop webcam when esc key is pressed"""
        cv2.destroyWindow("Sign Language Recognition")
        self.vc.release()

    def get_frame(self):
        """return the current frame from the webcam"""
        ret, frame = self.vc.read()
        if ret:
            print(ret)
            url = '/frames/frame.png'
            cv2.imwrite(os.getcwd() + '/frames/frame.png', frame)
            return url
        else:
            print(ret)
            return "no frame"

"""
cam = Webcam_Link(0)
print(cam.get_frame())
print(cam.get_frame())
"""


