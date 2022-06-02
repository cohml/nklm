from pathlib import Path
from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).resolve().parent

setup(
    name='nklm',
    version='1.0.0',
    description='Resources for building fake North Korean propaganda generators',
    long_description=(PROJECT_ROOT / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/cohml/nklm',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.10'
    ],
    entry_points={
        'console_scripts':
            [
                'scrape = nklm.collect.scrape:main',
            ]
        },
)
