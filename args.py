import argparse

def make_parser():
    """Makes a argument parser for some camera arguments

    Returns:
        argparse.ArgumentParser: The parser with the correct arguments
    """
    parser = argparse.ArgumentParser(description="Process the cmd arguments to the script")
    parser.add_argument("-c", "--cam", type=int, default=0, help="The camera id to use")
    parser.add_argument("-v", "--video", type=str, default=None, help="The video to use as the source for the capture")
    return parser