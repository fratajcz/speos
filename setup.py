from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='speos',
    version='0.2.1',
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
    packages=find_packages(
        where='.',
        include=['speos', 'speos/utils'],
        #exclude=['scripts', "tests"]
        ),
    #package_dir={"": "coregenes"},
    install_requires=[],
)
