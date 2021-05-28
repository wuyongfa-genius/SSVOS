from setuptools import setup, find_packages

setup(name='SSVOS',
      version='0.0.0',
      description='',
      author='wuyongfa, ly',
      author_email='',
      url='https://github.com/wuyongfa-genius/SSVOS.git',
      install_requires=['torchvision', 'timm', 'einops', 'accelerate', 'decord', 'spatial_correlation_sampler'],
      packages=find_packages()
      )