from setuptools import setup, find_packages

setup(name='human_hand_prior',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      author='Jongseob Yun',
      author_email='whdtjq4144@gmail.com',
      )