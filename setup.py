from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dropwise",
    version="0.2.1",
    author="Aryan Patil",
    author_email="aryanator01@gmail.com",
    description="Monte Carlo Dropout-based uncertainty estimation for Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aryanator/dropwise",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7',
)
