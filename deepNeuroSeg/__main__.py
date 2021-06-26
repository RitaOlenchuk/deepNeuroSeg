import click
from deepNeuroSeg import SegmentationFactory, SegmentationType

@click.command()
@click.option('--type', type=click.Choice(['wmh', 'c'], case_sensitive=True),  help="Either 'wmh' (White Matter Hyperintensities) or 'c' (Claustrum)")
@click.option('--flair', help="Path to nii.gz file with a FLAIR scan.", required=False, default=None, type=click.Path(exists=True))
@click.option('--t1', help="Path to nii.gz file with a T1 scan.", required=False, default=None, type=click.Path(exists=True))
@click.option('--o', help="Path where to save the resulting segmentation. Directory path or specific nii.gz file path.", required=True)
def run(type, flair, t1, o):
    if type=='wmh':
        if flair is None:
            raise TypeError('FLAIR scan is needed for \'wmh\' (White Matter Hyperintensities) Segmentation.')
        if not flair.endswith('.nii.gz'):
            raise NameError('Invalide FLAIR file expension. Must end with .nii.gz')
        if t1 is not None and not t1.endswith('.nii.gz'):
            raise NameError('Invalide T1 file expension. Must end with .nii.gz')
        segmenter = SegmentationFactory.create_segmenter(SegmentationType.WMH, FLAIR_path=flair, T1_path=t1)
        _ = segmenter.perform_segmentation(outputPath=o)
    elif type=='c':
        if t1 is None:
            raise TypeError('T1 scan is needed for \'c\' (Claustrum) Segmentation.')
        if not t1.endswith('.nii.gz'):
            raise NameError('Invalide T1 file expension. Must end with .nii.gz')
        segmenter = SegmentationFactory.create_segmenter(SegmentationType.Claustrum, T1_path=t1)
        _ = segmenter.perform_segmentation(outputPath=o)
