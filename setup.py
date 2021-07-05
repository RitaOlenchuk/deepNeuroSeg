from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='deepNeuroSeg',
    packages = ['deepNeuroSeg'],
    version='v0.4',
    license='MIT', 
    author='Margaryta Olenchuk',
    description='Deep-learning Tool for White Matter (WM) lesions and Claustrum structure segmentation in brain magnetic resonance imaging (MRI).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = 'https://github.com/RitaOlenchuk/deepNeuroSeg',
    download_url = 'https://github.com/RitaOlenchuk/deepNeuroSeg/archive/refs/tags/v0.4.tar.gz',
    keywords = ['deep-learning', 'machine-learning', 'segmentation', 'MRI', 'White Matter Lesions', 'Claustrum'],
    python_requires='>=3.9',
    install_requires=[
        'numpy==1.19.5',
        'tensorflow==2.5.0-rc3',
        'keras==2.5.0rc0',
        'SimpleITK==2.0.2',
        'scipy==1.6.3',
        'pytest==6.2.4',
        'click==8.0.0',
        'Pillow==8.2.0',
    ],
    entry_points={
        'console_scripts': [
            'deepNeuroSeg = deepNeuroSeg.__main__:run',
        ]
    }
)
