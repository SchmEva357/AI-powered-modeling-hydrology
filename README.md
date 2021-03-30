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
echo http://127.0.0.1:5000/ > .mlflow_uri
```

this will create a local file where the uri is stored. Alternatively you can export it as an environment variable with

```bash
export MLFLOW_URI=http://127.0.0.1:5000/
```

This links to your local mlflow, if you want to use a different one, then change the set uri.

The code in the [config.py](modeling/config.py) will try to read it locally and if the file doesn't exist will look in the env var.. IF that is not set the URI will be empty in your code.

## Usage

### Creating an MLFlow experiment

You can do it via the GUI or via [command line](https://www.mlflow.org/docs/latest/tracking.html#managing-experiments-and-runs-with-the-tracking-service-api) if you use the local mlflow:

```bash
mlflow experiments create --experiment-name 0-template-ds-modeling
```

Check your local mlflow

```bash
mlflow ui
```

and open the link [http://127.0.0.1:5000](http://127.0.0.1:5000)

This will throw an error if the experiment already exists. Save the experiment name in the [config file](modeling/config.py)

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python -m modeling.train
```

In order to test that predict works on a test set you created run:

```bash
python modeling/predict.py models/linear data/X_test.csv data/y_test.csv
```

## Limitations

development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible
