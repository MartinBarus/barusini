import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="barusini",
    version="0.0.1",
    author="Martin Barus",
    author_email="martin.barus@gmail.com",
    description="Automated ML pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MartinBarus/barusini",
    project_urls={
        "Bug Tracker": "https://github.com/MartinBarus/barusini/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "barusini"},
    packages=setuptools.find_packages(where="barusini"),
    python_requires=">=3.6",
)
