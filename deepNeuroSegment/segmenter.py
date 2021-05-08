from abc import ABC, abstractmethod
from enum import Enum
import wmh_segmenter
import claustrum_segmenter
import load_models


class SegmentationType(Enum):
    WMH = 1
    Claustrum= 2


class SegmentationFactory:
    @staticmethod
    def create_segmenter(segmentationType: SegmentationType, **kwargs) -> "Segmentation":
        if segmentationType == SegmentationType.WMH:
            segmentation = WMHSegmentation(**kwargs)
        elif segmentationType == SegmentationType.Claustrum:
            segmentation = ClaustrumSegmentation(**kwargs)
        else:
            raise NotImplemented("Sorry :(")

        segmentation.load_model()
        return segmentation

class Segmentation(ABC):
    @abstractmethod
    def perform_segmentation(self):
        pass
    
    def load_model(self):
        dir_name, urls = self._get_links()
        model = load_models.get_from_cache(dir_name, urls)

    @abstractmethod
    def _get_links(self):
        pass
    
class WMHSegmentation(Segmentation):
    wmh_dict = {"pretrained_FLAIR_only": {'0.h5':'1RRHtM0P_9o3OrkaE99RnUOpTGvJXSwfF',
                                          '1.h5':'1-W1OpQX1NbHYvu9MPEmp7KXEdzqF5LOJ',
                                          '2.h5':'1PvG_mOpa8Nnu_PDJyBO5nmM23-b7mdOR'}, 
                "pretrained_FLAIR_T1": {'0.h5':'12WGZfHxPcd2zLySGG90FwK2ZBwCPWonJ',
                                        '1.h5':'1mcnMzOHTdc4GaUhMGay2aUqjnJwNK-lO',
                                        '2.h5':'1szKGCFVnpHbFuyn-XYk2WMWywwwkZ9kz'}}

    def __init__(self, FLAIR_path, T1_path=None):
        self.FLAIR_path = FLAIR_path
        self.T1_path = T1_path

    def perform_segmentation(self, outputDir=None):
        return wmh_segmenter.wmh_segmentation(FLAIR_path=self.FLAIR_path, T1_path=self.T1_path, outputDir=outputDir)

    def save_segmentation(self, original_pred, outputDir):
        wmh_segmenter.save_wmh_segmentation(original_pred=original_pred, FLAIR_path=self.FLAIR_path, outputDir=outputDir)

    def _get_links(self):
        if self.T1_path:
            return 'pretrained_FLAIR_T1', WMHSegmentation.wmh_dict['pretrained_FLAIR_T1']
        else:
            return 'pretrained_FLAIR_only', WMHSegmentation.wmh_dict['pretrained_FLAIR_only']

class ClaustrumSegmentation(Segmentation):

    def perform_segmentation(self):
        claustrum_segmenter.claustrum_segmentation()

def main():
    concrete_strategy_a = SegmentationFactory.create_segmenter(SegmentationType.WMH,
        FLAIR_path='/Users/rita/Uni/pmsd/deepNeuroSegment/deepNeuroSegment/input_dir/FLAIR.nii.gz',
        T1_path='/Users/rita/Uni/pmsd/deepNeuroSegment/deepNeuroSegment/input_dir/T1.nii.gz')
    pred_image = concrete_strategy_a.perform_segmentation()
    concrete_strategy_a.save_segmentation(original_pred=pred_image, outputDir='/Users/rita/Uni/pmsd/deepNeuroSegment/deepNeuroSegment/result')

if __name__ == "__main__":
    main()