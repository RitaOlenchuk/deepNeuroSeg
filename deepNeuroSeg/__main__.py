import click
from deepNeuroSeg import SegmentationFactory, SegmentationType

@click.command()
@click.option('--type', type=click.Choice(['wmh', 'c'], case_sensitive=True),  help="Either 'wmh' (White Matter Hyperintensities) or 'c' (Claustrum)")
@click.option('--flair', help="Path to .nii.gz file of a FLAIR scan.", required=True)
@click.option('--t1', help="Path to .nii.gz file of a T1 scan.", required=False, default=None)
@click.option('--o', help="Directory path where to save the resulting segmentation.", required=True)
def run(type, flair, t1, o):
    if type=='wmh':
        segmenter = SegmentationFactory.create_segmenter(SegmentationType.WMH, FLAIR_path=flair, T1_path=t1)
        _ = segmenter.perform_segmentation(outputDir=o)
    elif type=='c':
        SegmentationFactory.create_segmenter(SegmentationType.Claustrum)
