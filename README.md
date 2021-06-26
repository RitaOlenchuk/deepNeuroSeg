# deepNeuroSeg
There are two different segmentation tasks you can perform with deepNeuroSeg: White Matter Hyperintensities (WMH) and Claustrum Segmentation. To learn details about the used deep-learning models please refer to the publications [1] and [2]. As soon as the underlying models are required, they will be downloaded to your computer and placed in ~/.deepNeuroSeg folder.

## White Matter Hyperintensities (WMH) Segmentation
WMH segmentation can be performed either using FLAIR only or both FLAIR and T1 scans. The results would be similar to our experience [1], but the default is using both of them.

![](https://github.com/RitaOlenchuk/deepNeuroSeg/blob/main/images/wmh_example.png?raw=true)
Fig.1: Segmentation result on Singapore 34 [1]. From left to right: FLAIR MR image, the associated ground truth, segmentation result using FLAIR modality only and segmentation result using FLAIR and T1 modalities. In column SegF LAIR and SegF LAIR+T1, the green area is the overlap between the segmentation maps and the ground-truth, the red pixels are the false negatives and the black ones are the false positives.

## Claustrum Segmentation
For Claustrum Segmentation the T1 scan must be provided. 

# How to:
Run deepNeuroSeg either in command line or Python.
## Command line
deepNeuroSeg performs two different segmentation tasks. The desired task must be specified with <code>--type</code> that is either equals to <code>wmh</code> (White Matter Hyperintensities) or <code>c</code> (Claustrum). For example:
```bash
deepNeuroSeg --type wmh --flair YOUR_PATH.nii.gz --t1 YOUR_PATH.nii.gz --o YOUR_PATH
```
Or: 
```bash
deepNeuroSeg --type c --t1 YOUR_PATH.nii.gz --o YOUR_PATH
```

For more details see:
```text
deepNeuroSeg --help
Options:
  --type [wmh|c]  Either 'wmh' (White Matter Hyperintensities) or 'c'
                  (Claustrum)
  --flair PATH    Path to nii.gz file with a FLAIR scan.
  --t1 PATH       Path to nii.gz file with a T1 scan.
  --o TEXT        Path where to save the resulting segmentation. Directory path or specific nii.gz file path.
                  [required]
  --help          Show this message and exit.
```
The resulting mask will be saved with user-specified .nii.gz file name or in the user-specified directory under the name out_mask.nii.gz.

## Python
In Python user will have to follow the next steps:
1. Import <code>deepNeuroSeg</code>
```python
from deepNeuroSeg import SegmentationFactory, SegmentationType
```
2. Create a <code>SegmentationFactory</code> object with segmentation type either <code>SegmentationType.Claustrum</code> or <code>SegmentationType.WMH</code>. An example for WMH Segmentation with both FLAIR and T1 modalities:
```python
segmenter = SegmentationFactory.create_segmenter(SegmentationType.WMH, 
                                                    FLAIR_path='YOUR_PATH',
                                                    T1_path='YOUR_PATH')
```
Or claustrum segmentation:
```python
segmenter = SegmentationFactory.create_segmenter(SegmentationType.Claustrum, 
                                                  T1_path='YOUR_PATH')
```

3. Next the segmentation can be performed.
   
Option 1: The user can specify the output path directly in <code>perform_segmentation</code> method.
```python
prediction = segmenter.perform_segmentation(outputDir='YOUR_PATH')
```
Option 2: The output numpy array can be inspected first, and then saved with <code>save_segmentation</code> function.

```python
prediction = segmenter.perform_segmentation()
segmenter.save_segmentation(mask=prediction, outputDir='YOUR_PATH')
```
In both cases, the output mask will be saved with user-specified .nii.gz file name or in user-specified directory under the name out_mask.nii.gz.

**Special feature of Claustrum Segmentation:**

The user can check the orientation of the coronal and axial images by selecting the special feature in <code>perform_segmentation</code> method:
```python
prediction = segmenter.perform_segmentation(check_orientation=True)
```
<code>check_orientation=True</code> will save the images of coronal and axial slices under ~/.deepNeuroSeg/images/.


# References:

[1]: Li, Hongwei, et al. "Fully convolutional network ensembles for white matter hyperintensities segmentation in MR images." NeuroImage 183 (2018): 650-665.

[2]: Li, Hongwei, et al. "Complex Grey Matter Structure Segmentation in Brains via Deep Learning: Example of the Claustrum." arXiv preprint arXiv:2008.03465 (2020).