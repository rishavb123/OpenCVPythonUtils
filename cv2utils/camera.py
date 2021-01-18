"""A file for the camera class"""
import cv2
import time
import threading
import pyvirtualcam
import numpy as np

from .constants import sample_length
from .args import make_parser


class Camera:
    """Camera class for streaming and taking pictures either from a video or webcam"""

    def __init__(
        self,
        src=0,
        name="Frame",
        should_log=True,
        frame_lock=None,
        res=(9999, 9999),
        fps=999,
    ):
        """Creates a Camera object with the parameters as settings

        Args:
            src (int or str, optional): The source for the camera: can be a video file or a camera id. Defaults to 0.
            name (str, optional): The name of the camera. Defaults to 'Frame'.
            should_log (bool, optional): Whether or not to log some basic information. Defaults to True.
            frame_lock (threading.Lock, optional): The lock object if you would like to access the cur_frame in another thread. Defaults to None.
            res (tuple, optional): The resolution the camera with sample with. Defaults to the max value of the camera you are using.
            fps (int, optional): The frames per second for the camera to sample with. Defaults to the max value of the camera you are using.
        """
        self.name = name
        self.src = src
        self.should_log = should_log
        self.cur_frame = None
        self.lock = frame_lock

        self.cap = src
        if not isinstance(self.cap, cv2.VideoCapture) and not isinstance(self.cap, Camera.Monitor):
            self.cap = cv2.VideoCapture(self.cap)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_res(self):
        """Gets the resolution of the internal opencv capture object

        Returns:
            tuple: the resolution in the format (w, h)
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def get_fps(self):
        """Gets the fps of the internal opencv capture object

        Returns:
            int: the fps
        """
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def set_lock(self, lock):
        """Sets the internal threading lock for self.cur_frame

        Args:
            lock (threading.Lock): the lock to set it to
        """
        self.lock = lock

    def set_res(self, res):
        """Sets the resolution of the internal opencv capture object

        Args:
            res (tuple): the resolution to set it to; (w,h)  is the format
        """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

    def set_fps(self, fps):
        """Sets the fps of the internal opencv capture object

        Args:
            fps (int): the fps to set it to
        """
        self.cap.set(cv2.CAP_PROP_FPS, fps)

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
        preprocess=lambda frames, raw: frames[0],
        output=None,
        frames_stored=1,
        prepare=lambda frame: cv2.flip(frame, 1),
        log=lambda fps=0, ret=True: print(f"\rFPS: {fps}", end=""),
        fps_sample_length=sample_length,
        finish=print,
    ):
        """Streams the camera output into a function or displays it

        Args:
            preprocess (function, optional): Function to pass the frames through before streaming it. Defaults to lambda frames:frame[0].
            output (function, optional): The function to output the preprocessed frame. Should receive list of frames and then return False to end the stream. Defaults to None.
            frames_stored (int, optional): The number of frames to pass into the preprocess function. Defaults to 1
            prepare (function, optional): The function to run on a frame before adding it to the buffer. Defaults to flipping it accross the y-axis.
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
            ret, raw = self.cap.read()
            if not ret:
                continue

            frame = prepare(raw)
            frames.insert(0, frame)
            if len(frames) <= frames_stored:
                continue
            frames.pop()
            output_frame = preprocess(frames=frames, raw=raw)

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
        preprocess=lambda frames, raw: frames[0],
        frames_stored=1,
        prepare=lambda frame: cv2.flip(frame, 1),
        log=lambda fps=0, ret=True: print(f"\rFPS: {fps}", end=""),
        fps_sample_length=sample_length,
        webcam_res=(None, None),
        finish=print,
    ):
        """Streams the preprocessed camera output into a virtual webcam device

        Args:
            preprocess (function, optional): Function to pass the frames through before streaming it. Defaults to lambda frames:frame[0].
            frames_stored (int, optional): The number of frames to pass into the preprocess function. Defaults to 1
            prepare (function, optional): The function to run on a frame before adding it to the buffer. Defaults to flipping it accross the y-axis.
            log (function, optional): The log function to pass the fps and ret values into. Defaults to lambda fps=0:print(f"\rFPS: {fps}", end='').
            fps_sample_length (int, optional): The sample length for the fps (if sample length is 10 it averages the last 10). Defaults to sample_length from constants.
            webcam_res (tuple, optional): The resolution of the webcam. Defaults to the resolution of the internal capture object.
            finish (function, optional): The function to run once the stream closes. Defaults to print.
        """
        currentFrame = 0
        frames = []

        last_time = time.time()
        times = []

        webcam_res = (
            webcam_res[0] or int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            webcam_res[1] or int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        with pyvirtualcam.Camera(
            width=webcam_res[0],
            height=webcam_res[1],
            fps=int(self.cap.get(cv2.CAP_PROP_FPS)),
        ) as cam:
            print("Press Control C to stop")
            while True:
                try:
                    ret, raw = self.cap.read()
                    if not ret:
                        continue

                    frame = prepare(raw)

                    frames.insert(0, frame)
                    if len(frames) <= frames_stored:
                        continue
                    frames.pop()
                    output_frame = preprocess(frames=frames, raw=raw)

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

                    cam.send(
                        cv2.resize(
                            cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGBA), webcam_res
                        )
                    )
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

    class Monitor:
        """A class to encapsulate the parameters to start reading pixels from the screen"""

        def __init__(
            self, sct, mon=0, top=0, left=0, width=1, height=1, updating=False
        ):
            """Contructs a Camera.Monitor object using the parameters passed in

            Args:
                sct (mss.mss): A reference to the mss.mss() object created outside the class
                mon (int, optional): Which monitor number to use. Defaults to 0.
                top (int, optional): The top position to capture. Defaults to 0.
                left (int, optional): The left position to capture. Defaults to 0.
                width (int, optional): The width to capture. Defaults to 1.
                height (int, optional): The height to capture. Defaults to 1.
                updating (bool, optional): Whether or not these values will update over time during the program. Defaults to False.
            """
            self.sct = sct
            self.mon = mon
            self.top = top
            self.left = left
            self.width = width
            self.height = height
            self.updating = updating
            self.__data = None
            m = self.sct.monitors[self.mon]

            def set_width(x):
                self.width = x / m["width"]

            def set_height(y):
                self.height = y / m["height"]

            self.__internal = {
                cv2.CAP_PROP_FRAME_WIDTH: {
                    "get": lambda: int(m["width"] * self.width),
                    "set": set_width,
                },
                cv2.CAP_PROP_FRAME_HEIGHT: {
                    "get": lambda: int(m["height"] * self.height),
                    "set": set_height,
                },
                cv2.CAP_PROP_FPS: {"get": lambda: 30, "set": lambda x: x},
            }

        def __json(self):
            if self.updating or self.__data is None:
                mon = self.sct.monitors[self.mon]
                self.__data = {
                    "top": int(mon["top"] + mon["height"] * self.top),
                    "left": int(mon["left"] + mon["width"] * self.left),
                    "width": int(mon["width"] * self.width),
                    "height": int(mon["height"] * self.height),
                }
            return self.__data

        def read(self):
            """Reads an image from the monitor defined in the contructor

            Returns:
                tuple: tuple containing the boolean value for whether the image capture worked and the captured iamge
            """
            try:
                return True, np.array(self.sct.grab(self.__json()))
            except:
                return False, None

        def get(self, code):
            """Gets a cv2 property of this capture object

            Args:
                code (int): the code from cv2

            Returns:
                object: The value at the specified code
            """
            return self.__internal[code]["get"]()

        def set(self, code, val):
            """Sets a cv2 property of this capture object

            Args:
                code (int): the code from cv2
                val (object): the value to set it to
            """
            self.__internal[code]["set"](val)


def make_camera_with_args(parser=None, **kwargs):
    """Creates a camera using the arguments passed in through command line and kwargs in this function

    Returns:
        tuple: the camera object created and the args parsed
    """
    parser = parser or make_parser()
    args = parser.parse_args()

    video = kwargs.get("video", args.video)
    cam = kwargs.get("cam", args.cam)
    should_log = kwargs.get("log", args.log)
    lock = kwargs.get("theading-lock", args.thread_lock)
    res = kwargs.get("res", args.resolution)
    fps = kwargs.get("fps", args.frames_per_second)

    camera = Camera(
        src=video if video else cam,
        should_log=should_log,
        frame_lock=threading.Lock() if lock else None,
        res=res,
        fps=fps,
    )
    return camera, args