from abc import ABC, abstractmethod
import wmh_segmenter
import claustrum_segmenter

class Context:
    def __init__(self, segmentation):
        self._segmentation = segmentation

    def context_interface(self):
        self._segmentation.perform_segmentation()

class Segmentation(ABC):
    @abstractmethod
    def perform_segmentation(self):
        pass

class WMHSegmentation(Segmentation):

    def perform_segmentation(self):
        wmh_segmenter.wmh_segmentation()

class ClaustrumSegmentation(Segmentation):

    def perform_segmentation(self):
        claustrum_segmenter.claustrum_segmentation()

def main():
    concrete_strategy_a = ClaustrumSegmentation()
    concrete_strategy_a.perform_segmentation()


if __name__ == "__main__":
    main()