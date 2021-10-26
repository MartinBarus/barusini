import pip
import setuptools

versions = tuple(int(x) for x in pip.__version__.split("."))
if versions >= (10, 0, 0):
    from pip._internal.req import parse_requirements
else:
    from pip.req import parse_requirements

requirements = parse_requirements("requirements.txt", session="requirements")
requirements = [x.requirement for x in requirements]


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
    project_urls={"Bug Tracker": "https://github.com/MartinBarus/barusini/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    packages=setuptools.find_packages(include=["barusini"]),
    package_data={"barusini": ["*/*.py", "*/*/*.py"]},
    python_requires=">=3.6",
)
