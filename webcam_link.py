"""
NAME: webcam_link.py
DESCRIPTION: use openCV to get camera access
PROGRAMMER: Caidan Gray
CREATION DATE: 4/3/2025
LAST EDITED: 4/4/2025   (please update each time the script is changed)
"""

import cv2


class Webcam_Link():
    def __init__(self, webcam: int):
        super(Webcam_Link, self).__init__()
        cv2.namedWindow("Not-Hotdog")
        self.vcc = None
        self.webcam = webcam

    def attempt_webcam_start(self):
        self.vc = cv2.VideoCapture(self.webcam)
        if self.vc.isOpened(): # try to get the first frame
            rval, frame = self.vc.read()
        else:
            rval = False

        while rval:
            cv2.imshow("preview", frame)
            rval, frame = self.vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                self.stop_webcam()

    def stop_webcam(self):
        cv2.destroyWindow("Not-Hotdog")
        self.vc.release()

cam = Webcam_Link(0)
# cam.attempt_webcam_start()