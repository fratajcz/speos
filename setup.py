from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='speos',
    version='0.2.3',
    author='Florin Ratajczak',
    author_email='florin.ratajczak@helmholtz-muenchen.de',
    description='We good to go',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fratajcz/speos",
    project_urls={
        "Bug Tracker": "https://github.com/fratajcz/speos/issues"
    },
    license='MIT',
    packages=find_packages(exclude=["speos.scripts","speos.tests"]),
    package_data={'': ['LICENSE.md', "speos/adjacencies.json", "speos/mapping.json", "speos/utils/config_default.yaml"]},
    #package_dir={"": "coregenes"},
    install_requires=["torch", "torch-geometric", "captum", "networkx", "h5py", "igraph", "matplotlib", "seaborn", "statsmodels", "scikit-learn", "pandas", "tensorboard", "pyyaml"],
)
