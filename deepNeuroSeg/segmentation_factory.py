from .segmentation_types import SegmentationType, AbstractSegmenter
from .wmh_segmenter import WMHSegmentation
from .claustrum_segmenter import ClaustrumSegmentation


class SegmentationFactory:
    @staticmethod
    def create_segmenter(segmentationType: SegmentationType, **kwargs) -> AbstractSegmenter:
        if segmentationType == SegmentationType.WMH:
            segmentation = WMHSegmentation(**kwargs)
        elif segmentationType == SegmentationType.Claustrum:
            segmentation = ClaustrumSegmentation(**kwargs)
        else:
            raise NotImplemented("Sorry :(")

        segmentation.load_model()
        return segmentation