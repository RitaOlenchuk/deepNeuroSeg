from .segmentation_types import AbstractSegmenter


class ClaustrumSegmentation(AbstractSegmenter):

    def perform_segmentation(self):
        claustrum_segmenter.claustrum_segmentation()

    def _get_links(self):
        pass
