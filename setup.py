# setup.py
from setuptools import setup, find_packages

setup(
    name="easyFL",  # Name of your package
    version="0.1",  # Package version
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[],  # Optional: List dependencies here
    include_package_data=True,  # Include data files from MANIFEST.in if needed
    author="Bryan Cora",
    description="A federated learning library",
    long_description=open("README.md").read(),  # Optional: Long description from README
    long_description_content_type="text/markdown",
    url="https://github.com/brinwiththevlin/easyFL",  # Optional: URL to the project's homepage
    classifiers=[  # Optional: Metadata classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
