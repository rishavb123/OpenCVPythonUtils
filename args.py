"""File to handle cmd arguments for the camera module"""
import argparse


def make_parser(**kwargs):
    """Makes a argument parser for some camera arguments

    Returns:
        argparse.ArgumentParser: The parser with the correct arguments
    """

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(
        description="Process the cmd arguments to the script"
    )
    parser.add_argument("-c", "--cam", type=int, default=0, help="The camera id to use")
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default=None,
        help="The video to use as the source for the capture",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str2bool,
        default=True,
        help="Whether or not to log information",
    )
    parser.add_argument(
        "-tl",
        "--thread-lock",
        type=str2bool,
        default=False,
        help="Whether or not to use a thread lock while updating the frame",
    )
    return parser