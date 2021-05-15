from setuptools import setup
setup(
    name='deepNeuroSeg',
    packages = ['deepNeuroSeg'],
    version='1.0.0',
    license='MIT', 
    author='Margaryta Olenchuk',
    description='Deep-learning Tool for White Matter (WM) lesions and Claustrum structure segmentation in brain magnetic resonance imaging (MRI).',
    url = 'https://github.com/RitaOlenchuk/deepNeuroSeg',
    download_url = 'https://github.com/RitaOlenchuk/deepNeuroSeg/archive/refs/tags/1.0.0.tar.gz',
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
    ],
    entry_points={
        'console_scripts': [
            'deepNeuroSeg = deepNeuroSeg.__main__:run',
        ]
    }
)
