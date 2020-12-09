"""A file for the camera class"""
import cv2
import time

from constants import sample_length


class Camera:
    """Camera class for streaming and taking pictures either from a video or webcam"""

    def __init__(self, src=0, name="Frame", should_log=True):
        """Creates a Camera object with the parameters as settings

        Args:
            src (int or str, optional): The source for the camera: can be a video file or a camera id. Defaults to 0.
            name (str, optional): The name of the camera. Defaults to 'Frame'.
            should_log (bool, optional): Whether or not to log some basic information. Defaults to True.
        """
        self.name = name
        self.src = src
        self.should_log = should_log

        self.cap = src
        if not isinstance(self.cap, cv2.VideoCapture):
            self.cap = cv2.VideoCapture(self.cap)

    def read(self):
        """Reads an image from the internal capture object

        Returns:
            tuple: tuple containing a boolean of whether the image is valid and the actual image
        """
        return self.cap.read()

    def stream(
        self,
        preprocess=lambda frame: frame,
        output=None,
        log=lambda fps=0, ret=True: print(f"\rFPS: {fps}", end=""),
        fps_sample_length=sample_length,
        finish=print,
    ):
        """Streams the camera output into a function or displays it

        Args:
            preprocess (function, optional): Function. Defaults to lambda frame:frame.
            output ([type], optional): [description]. Defaults to None.
            log ([type], optional): [description]. Defaults to lambdafps=0.
            ret ([type], optional): [description]. Defaults to False:print(f"\rFPS: {fps}", end='').
            fps_sample_length ([type], optional): [description]. Defaults to sample_length.
            finish ([type], optional): [description]. Defaults to print.
        """
        currentFrame = 0

        last_time = time.time()
        times = []

        while True:
            ret, frame = self.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = preprocess(frame)

            if output == None:
                cv2.imshow(self.name, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or k == 27:
                    break

            if self.should_log:
                currentFrame += 1
                cur_time = time.time()
                dt = cur_time - last_time
                times.append(dt)
                if len(times) > fps_sample_length:
                    times.pop(0)
                log(fps=len(times) / sum(times), ret=ret)
                last_time = cur_time

        finish()
        self.cap.release()
        cv2.destroyAllWindows()