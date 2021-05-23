# deepNeuroSeg
There are two different segmentation tasks you can perform: White Matter Lesions (WMH - White Matter Hyperintensities) and Claustrum Segmentation. The required models will be downloaded to ~/.deepNeuroSeg as soon as needed. WMH segmentation can be performed either using FLAIR only or both FLAIR and T1. The results would be similar to our experience [1]. The default is using both of them.

![](https://github.com/RitaOlenchuk/deepNeuroSeg/blob/main/images/wmh_example.png?raw=true)
Fig.1: Segmentation result on Singapore 34 [1]. From left to right: FLAIR MR image, the associated ground truth, segmentation result using FLAIR modality only and segmentation result using FLAIR and T1 modalities. In column SegF LAIR and SegF LAIR+T1, the green area is the overlap between the segmentation maps and the ground-truth, the red pixels are the false negatives and the black ones are the false positives.

Claustrum segmentation is still in development.

# How to:
Run deepNeuroSeg either in command line or Python.
### Command line
deepNeuroSeg performs two different segmentation tasks. The desired task must be specified with <code>--type</code> that is either equals to <code>wmh</code> (White Matter Hyperintensities (Lesions)) or <code>c</code> (Claustrum). For example:
```ruby
deepNeuroSeg --type wmh --flair YOUR_PATH.nii.gz --o YOUR_PATH
```
For more details see:
```ruby
deepNeuroSeg --help
Options:
  --type [wmh|c]  Either 'wmh' (White Matter Hyperintensities) or 'c'
                  (Claustrum)
  --flair TEXT    Path to .nii.gz file of a FLAIR scan.  [required]
  --t1 TEXT       Path to .nii.gz file of a T1 scan. [optional]
  --o TEXT        Directory path where to save the resulting segmentation.
                  [required]
```
The resulting mask will be saved in the user-specified directory under the name <code>out_mask.nii.gz</code>.
### Python
In Python user will have to follow the following steps by creating a <code>SegmentationFactory</code> object that can have a segmentation type either <code>SegmentationType.WMH</code> or <code>SegmentationType.Claustrum</code>. An example for WMH Segmentation with both FLAIR and T1 modalities:
```ruby
from deepNeuroSeg import SegmentationFactory, SegmentationType

segmenter = SegmentationFactory.create_segmenter(SegmentationType.WMH, 
                                                    FLAIR_path='YOUR_PATH',
                                                    T1_path='YOUR_PATH')
```
An example for WMH Segmentation with FLAIR only:
```ruby
from deepNeuroSeg import SegmentationFactory, SegmentationType

segmenter = SegmentationFactory.create_segmenter(SegmentationType.WMH, 
                                                    FLAIR_path='YOUR_PATH')
```
Then the segmentation can be performed. Here user can specify the output directory where the segmentation mask will be saved as <code>out_mask.nii.gz</code>:
```ruby
prediction = segmenter.perform_segmentation(outputDir='YOUR_PATH')
```
or inspect the numpy array yourself and save it with <code>save_segmentation</code> method.
```ruby
prediction = segmenter.perform_segmentation()
segmenter.save_segmentation(mask=prediction, outputDir='YOUR_PATH')
```
In both cases, the prediction can be saved in the output directory desired by the user under the name <code>out_mask.nii.gz</code>.

[1]: Li, Hongwei, et al. "Fully convolutional network ensembles for white matter hyperintensities segmentation in MR images." NeuroImage 183 (2018): 650-665.