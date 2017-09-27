import os
from setuptools import setup

setup(name='scsuite',
      version='0.1',
      description='single-cell Scalable Unified Inference of Trajectory Ensembles',
      author='Pablo Cordero',
      author_email='dimenwarper@gmail.com',
      license='MIT',
      url='https://github.com/dimenwarper/scsuite',
      packages=['scsuite'],
      entry_points={
            'console_scripts': ['scsuite = scsuite.main:top_level_command']
      },
      scripts=['scripts/scran.R', 'scripts/m3drop.R'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'networkx',
                        'matplotlib', 'pandas', 'seaborn', 'pyaml', 'colorama']
      )
print 'Installing some required R packages'
#os.system('R CMD BATCH install-packages.R')
