"""
NAME: webcam_link.py
DESCRIPTION: use openCV to get camera access
PROGRAMMER: Caidan Gray
CREATION DATE: 4/3/2025
LAST EDITED: 4/7/2025   (please update each time the script is changed)
"""

import cv2


class Webcam_Link():
    def __init__(self, webcam: int):
        """initialize the webcam link by creating variables and a named window"""
        super(Webcam_Link, self).__init__()
        # create a window
        cv2.namedWindow("Sign Language Recognition")
        self.vc = None
        self.webcam = webcam

    def attempt_webcam_start(self):
        """attempt to start the webcam"""
        self.vc = cv2.VideoCapture(self.webcam)
        # check that the window for the webcam is created
        if self.vc.isOpened(): # try to get the first frame
            rval, frame = self.vc.read()
        else:
            rval = False

        while rval:
            # put the webcam input in the window
            cv2.imshow("preview", frame)
            rval, frame = self.vc.read()
            # set key to esc key
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                self.stop_webcam()

    def stop_webcam(self):
        """stop webcam when esc key is pressed"""
        cv2.destroyWindow("Sign Language Recognition")
        self.vc.release()

    def get_frame(self):
        """return the current frame from the webcam"""
        return self.vc.read()

"""
cam.attempt_webcam_start()
cam = Webcam_Link(0)
"""