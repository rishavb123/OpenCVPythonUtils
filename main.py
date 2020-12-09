"""Main script"""
from camera import make_camera_with_args

camera, args = make_camera_with_args()
camera.stream()
