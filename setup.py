import pip
import setuptools

versions = tuple(int(x) for x in pip.__version__.split("."))
if versions >= (10, 0, 0):
    from pip._internal.req import parse_requirements
else:
    from pip.req import parse_requirements

requirements = parse_requirements("requirements.txt", session="requirements")
requirements = [x.requirement for x in requirements]

extras = {
    "tabular": ["optuna"],
    "nn": [
        "albumentations",
        "opencv-python",
        "pytorch_lightning",
        "timm",
        "torch",
        "transformers",
    ],
    "extra": ["lightgbm", "xgboost"],
}

extras_require = {"complete": []}

for package, reqs in extras.items():
    extras_require[package] = []
    for act_req in reqs:
        for req in requirements:
            if act_req in req:
                extras_require[package].append(req)
                extras_require["complete"].append(req)

    extras_require[package] = sorted(extras_require[package])

extras_require["complete"] = sorted(set(extras_require["complete"]))

requirements = sorted([x for x in requirements if x not in extras_require["complete"]])

packages = ["barusini"]


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
    packages=packages,
    package_data={"barusini": ["*/*.py", "*/*/*.py"]},
    extras_require=extras_require,
    python_requires=">=3.6",
)
