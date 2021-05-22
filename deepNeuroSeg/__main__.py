import click
from deepNeuroSeg import SegmentationFactory, SegmentationType

@click.command()
@click.option('--type', type=click.Choice(['wmh', 'c'], case_sensitive=True),  help="Either 'wmh' (White Matter Hyperintensities) or 'c' (Claustrum)")
@click.option('--flair', help="Path to .nii.gz file of a FLAIR scan.", required=True, type=click.Path(exists=True))
@click.option('--t1', help="Path to .nii.gz file of a T1 scan.", required=False, default=None, type=click.Path(exists=True))
@click.option('--o', help="Directory path where to save the resulting segmentation.", required=True)
def run(type, flair, t1, o):
    if type=='wmh':
        if not flair.endswith('.nii.gz'):
            raise NameError('Invalide FLAIR file expension. Must end with .nii.gz')
        if t1 is not None and not t1.endswith('.nii.gz'):
            raise NameError('Invalide T1 file expension. Must end with .nii.gz')
        segmenter = SegmentationFactory.create_segmenter(SegmentationType.WMH, FLAIR_path=flair, T1_path=t1)
        _ = segmenter.perform_segmentation(outputDir=o)
    elif type=='c':
        SegmentationFactory.create_segmenter(SegmentationType.Claustrum)
