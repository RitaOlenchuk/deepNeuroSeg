# deepNeuroSeg
# DeepNeuroSeq
There are two different segmentation tasks you can perform: White Matter Lesions (WMH - White Matter Hyperintensities) or Claustrum. For WMH segmentation can be performed either using FLAIR only or both FLAIR and T1. The results would be similar to our experience. The default is using both of them.
Claustrum segmentation is still in development.
# How to:
Run deepNeuroSeg either in command line or Python.
#### Command line
deepNeuroSeg performs two different segmentation tasks. The desired task must be specified with --type that is either equals to "wmh" (White Matter Hyperintensities (Lesions)) or "c" (Claustrum). For example:
```sh
deepNeuroSeg --type wmh --flair YOUR_PATH.nii.gz --o YOUR_PATH
```
For more details see:
```sh
deepNeuroSeg --help
Options:
  --type [wmh|c]  Either 'wmh' (White Matter Hyperintensities) or 'c'
                  (Claustrum)
  --flair TEXT    Path to .nii.gz file of a FLAIR scan.  [required]
  --t1 TEXT       Path to .nii.gz file of a T1 scan.
  --o TEXT        Directory path where to save the resulting segmentation.
                  [required]
```
The resulting mask will be saved in the user-specified directory under the name out_mask.nii.gz.
#### Python
In Python user will have to follow the following steps by creating a SegmentationFactory object that can have a segmentation type either SegmentationType.WMH or SegmentationType.Claustrum.
```sh
segmenter = SegmentationFactory.create_segmenter(SegmentationType.WMH, 
                                                    FLAIR_path='YOUR_PATH',
                                                    T1_path='YOUR_PATH')
```
Then the segmentation can be performed. Here user can specify the output directory where the segmentation mask will be saved as out_mask.nii.gz:
```sh
prediction = segmenter.perform_segmentation(outputDir='YOUR_PATH')
```
or inspect the numpy array yourself.
```sh
prediction = segmenter.perform_segmentation()
segmenter.save_segmentation(original_pred=prediction, outputDir='YOUR_PATH')
```
In both cases, the prediction can be saved in the output directory desired by the user under the name out_mask.nii.gz.