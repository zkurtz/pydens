from setuptools import setup

setup(
    name='pydens',
    version='0.1dev',
    packages=['pydens'],
    install_requires=[
        'pandas',
        'psutil',
        'scikit-learn',
        'shmistogram @ git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram'
    ],
    license='See LICENSE.txt'
)
