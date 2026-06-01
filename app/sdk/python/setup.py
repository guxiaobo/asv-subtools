"""Setup for the ASV SDK Python package."""
from setuptools import setup, find_packages

setup(
    name="asv-sdk",
    version="0.1.0",
    description="Python SDK for ASV Speaker Verification API",
    author="ASV-Subtools",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.27.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
