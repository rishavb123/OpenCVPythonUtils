import cv2
import time

from constants import sample_length

class Camera:

    def __init__(
        self, 
        src=0,
        name='Frame', 
        should_log=True
    ):
        self.name = name
        self.src = src
        self.should_log = should_log

        self.cap = src
        if not isinstance(self.cap, cv2.VideoCapture):
            self.cap = cv2.VideoCapture(self.cap)

    def read(self):
        return self.cap.read()

    def stream(
        self,
        preprocess=lambda frame: frame,
        output=None,
        log=lambda fps=0, ret=False: print(f"\rFPS: {fps}", end=''),
        fps_sample_length=sample_length,
        finish=print
    ):
        currentFrame = 0

        last_time = time.time()
        times = []

        while True:
            ret, frame = self.read()
            if not ret: continue

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