"""Setup script for CliffCast.

For most use cases, use pyproject.toml instead.
This file provides backwards compatibility for older pip versions.
"""

from setuptools import setup

# Read version from src/__init__.py
with open("src/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

setup(
    name="cliffcast",
    version=version,
    description="Transformer-based deep learning model for coastal cliff erosion prediction",
    author="Connor J. Mack",
    license="MIT",
    packages=["src"],
    python_requires=">=3.9",
)
