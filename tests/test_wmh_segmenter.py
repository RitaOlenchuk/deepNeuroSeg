import pytest

from deepNeuroSegment import SegmentationFactory, SegmentationType


def test_wmh_segmenter():
    concrete_strategy_a = SegmentationFactory.create_segmenter(SegmentationType.WMH,
        FLAIR_path='/Users/rita/Uni/pmsd/deepNeuroSegment/deepNeuroSegment/input_dir/FLAIR.nii.gz',
        T1_path='/Users/rita/Uni/pmsd/deepNeuroSegment/deepNeuroSegment/input_dir/T1.nii.gz')
    pred_image = concrete_strategy_a.perform_segmentation()
    concrete_strategy_a.save_segmentation(original_pred=pred_image, outputDir='/Users/rita/Uni/pmsd/deepNeuroSegment/deepNeuroSegment/result')
