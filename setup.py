from setuptools import setup, find_packages

__version__ = '0.0.1'
URL = None
install_requires = [
    "matplotlib",
    "networkx",
    "sklearn",
    "scipy",
    "jupyter",
    "jupyterlab",
    "tensorboard",
    "gensim", 
    'pandas'
]

setup(
    name='DIDA',
    version=__version__,
    description='DIDA',
    author='mnlab',
    url=URL,
    python_requires='>=3.8',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)

# pip install torch==1.11 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 

