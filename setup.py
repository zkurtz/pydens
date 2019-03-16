from setuptools import setup

exec(open("pydens/version.py").read())

version = {}
with open("pydens/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='pydens',
    version=version['__version__'],
    packages=['pydens', 'pydens.classifiers', 'pydens.models', 'pydens.simulators'],
    install_requires=[
        'fastkde',
        'lightgbm',
        'pandas',
        'psutil',
        'scikit-learn',
        'shmistogram @ git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram'
    ],
    license='See LICENSE.txt'
)
