# setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypbr",  # Your package name
    version="0.1.0a",
    author="Giuseppe Vecchio",
    author_email="giuseppevecchio@hotmail.com",
    description="A Python library for easy and fast manipulation of PBR materials with PyTorch integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giuvecchio/pypbr",  # Update with your repository URL
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.18.0",
        "Pillow>=8.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "License :: OSI Approved :: MIT License",  # Choose the appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
