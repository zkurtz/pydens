from setuptools import setup

exec(open("pydens/version.py").read())

version = {}
with open("pydens/version.py") as fp:
    exec(fp.read(), version)

try:
    import shmistogram
except:
    raise Exception("You first need to install shmistogram; try "
        + "`pip install git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram`")

try:
    import cython
except:
    raise Exception("You must first install Cython; `pip install Cython`")
setup(
    name='pydens',
    version=version['__version__'],
    packages=['pydens',
        'pydens.classifiers',
        'pydens.evaluation',
        'pydens.models',
        'pydens.simulators',
        'pydens.wrappers'
    ],
    install_requires=[
        'scikit-learn',
        # Installing fastke from source due to
        #   https://bitbucket.org/lbl-cascade/fastkde/issues/5/using-a-non-tuple-sequence-for
        'fastkde @ git+https://bitbucket.org/lbl-cascade/fastkde.git#egg=fastkde',
        'lightgbm',
        'pandas',
        'psutil',
        'shmistogram @ git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram'
    ],
    license='See LICENSE.txt'
)
