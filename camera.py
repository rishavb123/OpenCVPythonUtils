"""A file for the camera class"""
import cv2
import time

from constants import sample_length
from args import make_parser


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
        self.cur_frame = None

        self.cap = src
        if not isinstance(self.cap, cv2.VideoCapture):
            self.cap = cv2.VideoCapture(self.cap)

    def read(self):
        """Reads an image from the internal capture object

        Returns:
            tuple: tuple containing a boolean of whether the image is valid and the actual image
        """
        ret, frame = self.cap.read()
        if ret:
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
            ret, frame = self.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frames.insert(0, frame)
            if len(frames) <= frames_stored:
                continue
            frames.pop()
            output_frame = preprocess(frames)

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

    def _default_output_function(self, frame):
        cv2.imshow(self.name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            return False
        return True


def make_camera_with_args():
    parser = make_parser()
    args = parser.parse_args()

    camera = Camera(src=args.video if args.video else args.cam, should_log=args.log)
    return camera, args