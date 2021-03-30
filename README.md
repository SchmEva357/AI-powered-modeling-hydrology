# ds-modeling-pipeline

Skeleton project for building a simple model in python script
This is the simplest way to do it. We train a simple model in the jupyter notebook, where we select only some features and do minimal cleaning. The output is then stored in simple python scripts.

Data used is the [coffee quality dataset](https://github.com/jldbc/coffee-quality-database).

##

Requirements:

- pyenv with Python: 3.8.5

### Setup

Same procedure as last time...

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.8.5
python -m venv .venv
pip install --upgrade pip
pip install -r requirements_dev.txt
```

The requirements.txt file contains the libraries needed for deployment.. of model or dashboard .. thus no jupyter or other libs used during development.

The MLFLOW URI should not be stored on git, you have two options, to save it locally in the .mlflow_uri file:

```BASH
echo pasted_uri > .mlflow_uri
```

this will create a local file where the uri is stored. Alternatively you can export it as an environment variable with

```bash
export MLFLOW_URI=pasted_uri
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python modeling/train.py
```

In order to test that predict works on a test set you created run:

```bash
python modeling/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible
