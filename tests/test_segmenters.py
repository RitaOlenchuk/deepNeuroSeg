import pytest
import os
import deepNeuroSeg
from deepNeuroSeg import SegmentationFactory, SegmentationType

def test_SegmentationFactory_SegmentationTypeWMH():
    test_path = '../test_your_data_WMH/input_dir/FLAIR.nii.gz'
    concrete_strategy_a = SegmentationFactory.create_segmenter(SegmentationType.WMH,  FLAIR_path=test_path)
    assert isinstance(concrete_strategy_a, deepNeuroSeg.wmh_segmenter.WMHSegmentation)

def test_SegmentationFactory_SegmentationTypeClaustrum():
    test_path = '../Claustrum_Demo_GitHub/data/sub-0051456/sub-0051456_T1w_denoised.nii'
    concrete_strategy_b = SegmentationFactory.create_segmenter(SegmentationType.Claustrum,  T1_path=test_path)
    assert isinstance(concrete_strategy_b, deepNeuroSeg.claustrum_segmenter.ClaustrumSegmentation)

def test_SegmentationFactory_WNH_initialization():
    test_path = '../test_your_data_WMH/input_dir/FLAIR.nii.gz'
    concrete_strategy_a = SegmentationFactory.create_segmenter(SegmentationType.WMH, FLAIR_path=test_path)
    assert concrete_strategy_a.get_FLAIR_path() == test_path and concrete_strategy_a.get_T1_path() == None

def test_wmh_segmenter():
    concrete_strategy_a = SegmentationFactory.create_segmenter(SegmentationType.WMH,
                                                               FLAIR_path='../test_your_data_WMH/input_dir/FLAIR.nii.gz',
                                                               T1_path='../test_your_data_WMH/input_dir/T1.nii.gz')
    pred_image = concrete_strategy_a.perform_segmentation()
    test_path = '../test_your_data_WMH/result/test.nii.gz'
    concrete_strategy_a.save_segmentation(mask=pred_image, outputPath=test_path)
    assert os.path.exists(test_path)

def test_claustrum_segmenter():
    concrete_strategy_b = SegmentationFactory.create_segmenter(SegmentationType.Claustrum,
                                                               T1_path='/Users/rita/Uni/pmsd/Claustrum_Demo_GitHub/data/sub-0051456/sub-0051456_T1w_denoised.nii')
    _ = concrete_strategy_b.perform_segmentation(check_orientation=True)
    test_path = os.path.join( os.path.realpath(os.path.expanduser('~/.deepNeuroSeg')), 'images')
    assert os.path.exists(test_path) and not len(test_path) == 0