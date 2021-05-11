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

def main():
    concrete_strategy_a = SegmentationFactory.create_segmenter(SegmentationType.WMH,
        FLAIR_path='/Users/rita/Uni/pmsd/deepNeuroSeg/deepNeuroSeg/input_dir/FLAIR.nii.gz',
        T1_path='/Users/rita/Uni/pmsd/deepNeuroSeg/deepNeuroSeg/input_dir/T1.nii.gz')
    pred_image = concrete_strategy_a.perform_segmentation()
    concrete_strategy_a.save_segmentation(original_pred=pred_image, outputDir='/Users/rita/Uni/pmsd/deepNeuroSegment/deepNeuroSegment/result')

if __name__ == "__main__":
    main()