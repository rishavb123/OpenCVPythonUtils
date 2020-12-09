"""Main script"""
from args import make_parser
from camera import Camera

parser = make_parser()
args = parser.parse_args()

camera = Camera(src=args.video if args.video else args.cam)
camera.stream()
