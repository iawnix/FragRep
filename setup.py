from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
      name="FragRep"
    , version="1.0"
    , author="iawnix"
    , author_email="iawnix@163.com"
    , description="Substitution of functional groups from the core scaffold via a single bond."
    , install_requires=read_requirements()
    , packages=find_packages()
    , entry_points={
        "console_scripts": [
            "FragRep=src.run:main"]}
    , python_requires=">=3.12"
)
