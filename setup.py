from setuptools import setup, find_packages

setup(
    name="causalcellbench",
    version="1.0.0",
    description="Benchmarking virtual cell models for causal gene regulatory network discovery",
    author="Aayan Alwani",
    url="https://github.com/aayanalwani/causalcellbench",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "anndata>=0.10",
        "scanpy>=1.10",
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "networkx>=3.0",
        "matplotlib>=3.7",
        "gies>=0.0.3",
        "h5py>=3.8",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
