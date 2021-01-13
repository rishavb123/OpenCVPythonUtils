"""A file for the camera class"""
import cv2
import time
import threading
import pyvirtualcam
import numpy as np

from constants import sample_length
from args import make_parser


class Camera:
    """Camera class for streaming and taking pictures either from a video or webcam"""

    def __init__(self, src=0, name="Frame", should_log=True, frame_lock=None):
        """Creates a Camera object with the parameters as settings

        Args:
            src (int or str, optional): The source for the camera: can be a video file or a camera id. Defaults to 0.
            name (str, optional): The name of the camera. Defaults to 'Frame'.
            should_log (bool, optional): Whether or not to log some basic information. Defaults to True.
            frame_lock (threading.Lock, optional): The lock object if you would like to access the cur_frame in another thread. Defaults to None.
        """
        self.name = name
        self.src = src
        self.should_log = should_log
        self.cur_frame = None
        self.lock = frame_lock

        self.cap = src
        if not isinstance(self.cap, cv2.VideoCapture):
            self.cap = cv2.VideoCapture(self.cap)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
        self.cap.set(cv2.CAP_PROP_FPS, 999)

    def set_lock(self, lock):
        """Sets the internal threading lock for self.cur_frame

        Args:
            lock (threading.Lock): the lock to set it to
        """
        self.lock = lock

    def read(self):
        """Reads an image from the internal capture object

        Returns:
            tuple: tuple containing a boolean of whether the image is valid and the actual image
        """
        ret, frame = self.cap.read()
        if self.lock:
            with self.lock:
                self.cur_frame = frame
        else:
            self.cur_frame = frame
        return ret, frame

    def stream(
        self,
        preprocess=lambda frames: frames[0],
        output=None,
        frames_stored=1,
        log=lambda fps=0, ret=True: print(f"\rFPS: {fps}", end=""),
        fps_sample_length=sample_length,
        finish=print,
    ):
        """Streams the camera output into a function or displays it

        Args:
            preprocess (function, optional): Function to pass the frames through before streaming it. Defaults to lambda frames:frame[0].
            output (function, optional): The function to output the preprocessed frame. Should receive list of frames and then return False to end the stream. Defaults to None.
            frames_stored (int, optional): The number of frames to pass into the preprocess function. Defaults to 1
            log (function, optional): The log function to pass the fps and ret values into. Defaults to lambda fps=0:print(f"\rFPS: {fps}", end='').
            fps_sample_length (int, optional): The sample length for the fps (if sample length is 10 it averages the last 10). Defaults to sample_length from constants.
            finish (function, optional): The function to run once the stream closes. Defaults to print.
        """
        currentFrame = 0
        frames = []

        last_time = time.time()
        times = []

        if output == None:
            output = self._default_output_function

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frames.insert(0, frame)
            if len(frames) <= frames_stored:
                continue
            frames.pop()
            output_frame = preprocess(frames)

            if self.lock:
                with self.lock:
                    self.cur_frame = output_frame
            else:
                self.cur_frame = output_frame

            if output(output_frame) == False:
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

    def make_virtual_webcam(
        self,
        preprocess=lambda frames: frames[0],
        frames_stored=1,
        log=lambda fps=0, ret=True: print(f"\rFPS: {fps}", end=""),
        fps_sample_length=sample_length,
        finish=print,
    ):
        """Streams the preprocessed camera output into a virtual webcam device

        Args:
            preprocess (function, optional): Function to pass the frames through before streaming it. Defaults to lambda frames:frame[0].
            frames_stored (int, optional): The number of frames to pass into the preprocess function. Defaults to 1
            log (function, optional): The log function to pass the fps and ret values into. Defaults to lambda fps=0:print(f"\rFPS: {fps}", end='').
            fps_sample_length (int, optional): The sample length for the fps (if sample length is 10 it averages the last 10). Defaults to sample_length from constants.
            finish (function, optional): The function to run once the stream closes. Defaults to print.
        """
        currentFrame = 0
        frames = []

        last_time = time.time()
        times = []

        with pyvirtualcam.Camera(width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=int(self.cap.get(cv2.CAP_PROP_FPS))) as cam:
            print("Press Control C to stop")
            while True:
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)

                    frames.insert(0, frame)
                    if len(frames) <= frames_stored:
                        continue
                    frames.pop()
                    output_frame = preprocess(frames)

                    if self.lock:
                        with self.lock:
                            self.cur_frame = output_frame
                    else:
                        self.cur_frame = output_frame

                    if self.should_log:
                        currentFrame += 1
                        cur_time = time.time()
                        dt = cur_time - last_time
                        times.append(dt)
                        if len(times) > fps_sample_length:
                            times.pop(0)
                        log(fps=len(times) / sum(times), ret=ret)
                        last_time = cur_time
                        
                    cam.send(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGBA))
                    cam.sleep_until_next_frame()

                except KeyboardInterrupt as e:
                    finish()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break

    def _default_output_function(self, frame):
        cv2.imshow(self.name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            return False
        return True


def make_camera_with_args(**kwargs):
    """Creates a camera using the arguments passed in through command line and kwargs in this function

    Returns:
        tuple: the camera object created and the args parsed
    """
    parser = make_parser()
    args = parser.parse_args()

    video = kwargs.get("video", args.video)
    cam = kwargs.get("cam", args.cam)
    should_log = kwargs.get("log", args.log)
    lock = kwargs.get("theading-lock", args.thread_lock)

    camera = Camera(
        src=video if video else cam,
        should_log=should_log,
        frame_lock=threading.Lock() if lock else None,
    )
    return camera, args