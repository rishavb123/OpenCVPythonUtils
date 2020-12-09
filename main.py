from args import make_parser
from camera import Camera

parser = make_parser()
args = parser.parse_args()

camera = Camera()
camera.stream()
