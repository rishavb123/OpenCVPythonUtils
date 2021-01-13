import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cv2utils-rbhagat",
    version="1.1.0",
    author="Rishav Bhagat",
    author_email="rishav@bhagat.io",
    description="A package that has a bunch of utility functions and classes for opencv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rishavb123/OpenCVPythonUtils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
)