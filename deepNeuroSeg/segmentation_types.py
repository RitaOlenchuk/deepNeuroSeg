import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict
from .run_utils import run_script_inline

class SegmentationType(Enum):
    WMH = 1
    Claustrum= 2


class AbstractSegmenter(ABC):
    @abstractmethod
    def perform_segmentation(self):
        pass
    
    def load_model(self):
        dir_name, urls = self._get_links()
        AbstractSegmenter.get_from_cache(dir_name, urls)

    @abstractmethod
    def _get_links(self):
        pass

    @staticmethod
    def get_from_cache(dir_name: str, urls: Dict[str, str], cache_dir: str = "~/.deepNeuroSeg"):
        cache_dir = os.path.realpath(os.path.expanduser(cache_dir))
        print(cache_dir)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        onlydirs = [f for f in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, dir_name))]
        print(onlydirs)
        if (not dir_name in onlydirs) or len(os.listdir(os.path.join(cache_dir, dir_name)))==0:
            deeper_dir = os.path.join(os.path.join(cache_dir, dir_name))
            if not os.path.exists(deeper_dir):
                os.mkdir(deeper_dir)
            for key in urls:
                filename = os.path.join(deeper_dir, key)
                file_id = urls[key]
                run_script_inline(f"""
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={file_id}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id={file_id}" -o {filename}
rm ./cookie
""")