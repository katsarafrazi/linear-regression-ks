# Linear Regression Model 

## Introduction

Linear regression model module with model training using Statistitical Gradient Descent (SGD) or Gradient Descent. 

## Installation 

`pip install -i https://test.pypi.org/simple/ simple-linear-regression-ksarafrazi==1.0.0`

https://test.pypi.org/project/simple-linear-regression-ksarafrazi/1.0.0/

You can find installation files in the dist directory.

## System Overview

The below diagram shows an overview of the pipeline. Detailed explanations of each component is included next.

![overview.png](overview.png)

### Data Loader

The input data Diabetes dataset from sklearn datasets. This module reads the data.

### Data Validator

Validates if the input data is present and in the correct format. 
This part has been implemented using assertions in the data loader module and unittests. But, in an enterprise system with multiple data sources, this should be a standalone module.

### Data Transformer

Includes data wrangling and feature engineering. For this project, this module keeps the third feature in the dataset and reshapes the data. In a bigger project there would be separate modules for data cleaning, data transformation and feature selection. 

### Data Preprocessing

Includes preprocessing steps required for model training. For this project this only splits the data into training and test sets. But it could include feature scaling, one hot encoding or other required steps.

### Model Training

We use a Linear Regression model in this project. The user can choose the `learning rate`, `number of iterations` and `training mode`. For the `training mode` we have the option of `GD` (Gradient Descent) or `SGD` (Statistical Gradient Descent)
No hyper parameter optimization or cross validation is used, but in an enterprise pipeline we should include those steps.

The trained model is saved as a pickle file. Model object's `__predict` method can be called to make predictions.

### Model Evaluation

Evaluates model performance.

### API

The trained model is served as a Flask API. To create the API endpoint locally the user can run `main.py`
The API is also accessible publically on "https://ksarafrazi.pythonanywhere.com". The API endpoints include:

#### "/model-check" :
Check model health and information

#### "/stream" :
Returns prediction for a single input.

Sample API call
```
X = [0.06169621]
headers={'Content-type':'application/json'}
payload = {'data':json.dumps(X)}
y_predict = requests.post('https://ksarafrazi.pythonanywhere.com/stream', json=payload, headers=headers).json()
```

#### "/batch" :
Returns prediction for an array of inputs.

Sample API call
```
X = [[0.0616962065186885], [-0.0514740612388061], [0.0444512133365941]]
headers={'Content-type':'application/json'}
payload = {'data':json.dumps(X)}
y_predict = requests.post('https://ksarafrazi.pythonanywhere.com/batch', json=payload, headers=headers).json()
```

## Containerization
The code has been packaged into a docker container and can be accessed via:

```docker pull ksarafrazi/linear_regression```

## CI/CD Pipeline

Github actions is used to create a CI/CD pipeline.
The pipeline automatically lints the code, runs tests and builds a docker image. 
Unittests to test the data loader and API endpoint have been implemented in `src/src/simple-linear-regression-ksarafrazi/tests`. For an enterprise system more unittest and validator tests should be included.

Building and pushing a new docker image could potentially be automated as well, but it wasn't here since a paid DockerHub account was required.

## Notes
The provided equations for updating the weights seem to be for gradient descent, where all training samples are used in each iteration. Based on the definitions for `dW` and `db`, formulas should also be changed to `w = w + lr*dW` and `b = b + lr*db`.
I adjusted the formulas and implemented GD. I have also implemented SGD, which only uses one random sample to update the weights in each iteration.

If I had more time I would have added more tests, validators and monitoring modules. 
I also would have improved the training module by adding batched SGD and allowing early stopping. 