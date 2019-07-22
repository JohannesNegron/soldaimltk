import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="soldai-machine-learning-tool-kit",
    version="0.0.1",
    author="Soldai Research",
    author_email="juseng7@gmail.com",
    description="Soldai machine learning toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johannesnegron/soldaimltk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
