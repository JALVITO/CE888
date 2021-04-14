# Assignment 2

Repo containing code and examples for CE888 Assignment 2.

## Directory
* `notebooks`: contains jupyter notebooks with examples on how to use the implemented code.
* `saved_models`: contains the saved tensorflow models obtained during training. (_EDIT:_ Only best performing model has been kept due to git size upload limitations)
* `src`: contains the python code that implements the models and utility functions for data loading and augmenting.
* `tests`: contains unit tests for the implemented source code.

## How to run
1. Install the required libraries.
2. Set the desired hyper-parameters and update the paths in `src/config.py`
3. Build the data using `data_split()` from `src/data.py`.
4. Instantiate a `FireModel` (defined in `src/model.py`) as indicated in the example notebooks. The model can then be trained and evaluated.

> **Note:** Depending on the environment, specific care may be needed to properly load the implemented code into a notebook (such as appending the src folder to the path). See the `notebooks/TrainingExample.ipynb` for an example.

## FLAME Dataset
The dataset used to train and evaluate the models can be downloaded [here](https://drive.google.com/drive/folders/1Y46kOUage8tYd3LSCpaUaYXXhAnMNzb3?usp=sharing).