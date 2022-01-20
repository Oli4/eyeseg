# octseg

## Installation

## Usage

## Development
To get started clone this repository.

### Install dependencies

This project uses Poetry for dependency management and packaging. You can install with the following command:
```
curl -sSL https://install.python-poetry.org | python3 -
```
You might have to add `poetry` to your `PATH`. Then you can run `poetry install` from within this project to install it in editable mode with all dependencies.

You can find the documentation of Poetry [here](https://python-poetry.org/docs/). I mainly use it to add dependencies with `poetry add` or to start the project specific virtual environment `poetry shell`.

### Commit your changes

Before committing for the first time please install the pre-commit hooks with
```
pre-commit install
```
`pre-commit` is a development dependency of this project so it is already installed in the projects virtual environment. You can find the `pre-commit` documentation [here](https://pre-commit.com/)

One of the installed pre-commit hooks checks whether your commit message follows the [conventional commits](https://www.conventionalcommits.org/) standard. Standardized commit messages allow for the automatic creation of a changelog and automatic version bumping.
To create such a standardized commit message use `commitizen` to make your commit. It creates your commit message from a number of questions.
Commitizen is already installed in the project virtual environment and can be used with the following command:
```
cz commit
```

### Create a new version and publish to PyPI
If you
