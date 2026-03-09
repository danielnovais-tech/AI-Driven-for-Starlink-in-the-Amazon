from setuptools import setup, find_packages

setup(
    name="ai_starlink_amazon",
    version="0.1.0",
    description=(
        "Deep Reinforcement Learning for real-time beamforming optimization "
        "in LEO satellite constellations over the Amazon rainforest."
    ),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "gymnasium>=0.29.0",
        "h5py>=3.9.0",
    ],
    extras_require={
        "gnn": ["torch-geometric>=2.4.0"],
        "dev": ["pytest>=7.4.0"],
    },
)
