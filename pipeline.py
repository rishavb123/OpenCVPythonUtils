"""A file to hold the pipeline class"""


class Pipeline(list):
    """Class for creating a pipeline to process images"""

    def __init__(self, lst=None):
        """Inits a pipeline object

        Args:
                lst (list, optional): Initializes the pipeline with whatever is in the list. Defaults to None.
        """
        super().__init__()
        if lst is not None:
            super().extend(lst)

    def __call__(self, frame):
        """Calls all the functions stored in the pipeline

        Args:
                frame (np.array): An cv2 image object to process

        Returns:
                np.array: The processed image
        """
        for step in self:
            frame = step(frame)
        return frame
