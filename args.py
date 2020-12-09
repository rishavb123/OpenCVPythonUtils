import argparse

def make_parser():
    parser = argparse.ArgumentParser(description="Process the cmd arguments to the script")
    parser.add_argument("-c", "--cam", type=int, default=0, help="The camera id to use")
    parser.add_argument("-v", "--video", type=str, default=None, help="The video to use as the source for the capture")
    return parser