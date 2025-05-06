from setuptools import setup, find_packages
import os

# Чтение README.md для long_description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kan_fer",
    version="0.1.0",
    author="Ivan Lyschev",
    author_email="your.email@example.com",
    description="Распознавание эмоций с использованием KAN (Kolmogorov-Arnold Networks)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kan_fer_models",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "efficient-kan @ git+https://github.com/Blealtan/efficient-kan.git",
        "torchkan @ git+https://github.com/1ssb/torchkan.git",
    ],
    include_package_data=True,
    package_data={
        "kan_fer.pretrained": ["*.pt"],
    },
) 