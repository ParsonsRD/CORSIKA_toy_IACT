from setuptools import setup

setup(
    name='corsika_toy_iact',
    version='1.0',
    packages=['corsika_toy_iact'],
    package_dir={'corsika_toy_iact': 'corsika_toy_iact'},
    package_data={'corsika_toy_iact': ["data/test_data.root"]},
    include_package_data=True,
    url='',
    license='',
    author='dparsons',
    author_email='',
    description='Simple code for reproducing behaviour of Cherenkov telescopes'
)
