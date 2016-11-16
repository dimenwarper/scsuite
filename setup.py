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
      install_requires=['numpy', 'scipy', 'scikit-learn', 'networkx',
                        'matplotlib', 'pandas', 'seaborn', 'pyaml', 'colorama']
      )
