from .segmentation_types import SegmentationType, AbstractSegmenter
from .wmh_segmenter import WMHSegmentation
from .claustrum_segmenter import ClaustrumSegmentation


class SegmentationFactory:
    @staticmethod
    def create_segmenter(segmentationType: SegmentationType, **kwargs) -> AbstractSegmenter:
        """Initializes segmenter of the required type and parameters. In case of WMH Segmenter the FLAIR file path (.nii.gz) needs to be specified, optionaly the T1 file path (.nii.gz) can be provided as well.

        Args:
            segmentationType (SegmentationType): Either SegmentationType.WMH or SegmentationType.Claustrum corresponding to White Matter Hyperintensities or Claustrum Segmentation respectively.

        Raises:
            NotImplemented: in case SegmentationType is neither WMH nor Claustrum. 

        Returns:
            AbstractSegmenter: object that can be then used to perform the actual segmentation.
        """
        if segmentationType == SegmentationType.WMH:
            segmentation = WMHSegmentation(**kwargs)
        elif segmentationType == SegmentationType.Claustrum:
            segmentation = ClaustrumSegmentation(**kwargs)
        else:
            raise NotImplemented("Sorry :(")

        segmentation.load_model()
        return segmentation