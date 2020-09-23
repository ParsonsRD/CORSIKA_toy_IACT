from setuptools import setup, find_packages

setup(
    name='corsika_toy_iact',
    version='1.1',
    packages=find_packages(include=['corsika_toy_iact', 'corsika_toy_iact.*']),
    package_dir={'corsika_toy_iact': 'corsika_toy_iact'},
    package_data={'corsika_toy_iact': ["data/test_data.root"]},
    include_package_data=True,
    url='',
    license='',
    zip_safe=False,
    author='parsonsrd',
    description='Simple code for reproducing behaviour of Cherenkov telescopes',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
