import sys
import os

PACKAGE_PARENT = '../cv2utils'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import cv2
from flask import Response, Flask, render_template

from camera import make_camera_with_args

app = Flask(__name__)
camera = make_camera_with_args()

@app.route("/")
def index():
	return render_template("index.html")

def generate():
	global camera

	while True:
		with camera.lock:
			if camera.cur_frame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", camera.cur_frame)
			if not flag:
				continue
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')