from setuptools import setup, find_packages

setup(
    name='pybullet_env',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'pybullet',
        'hydra-core',
        'scipy',
        'PyYAML',
        'opencv-python',
    ],
    description='PyBullet environment for simulations',
    python_requires='>=3.9',
    author='Kallol Saha',
)
