from setuptools import setup

exec(open("pydens/version.py").read())

setup(
    name='pydens',
    version=__version__,
    packages=['pydens', 'pydens.classifiers', 'pydens.models', 'pydens.simulators'],
    install_requires=[
        'pandas',
        'psutil',
        'scikit-learn',
        'shmistogram @ git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram'
    ],
    license='See LICENSE.txt'
)
