# Linear Regression Model 

## Introduction

Linear regression model module with model training using Statititical Gradient Descent (SGD) or Gradient Descent. 

## Installation 
You can find installation files in the dist directory.

## System Overview

The below diagram shows an overview of the pipeline. Detailed explanations of each component is included next.

![overview.png](overview.png)

### Data Loader

The input data Diabetes dataset from sklearn datasets. This modeule reads the data.

### Data Validator

Validates if the input data is present and in the correct format. 
This part has been implemented using assertions in the data loader module and unittests. But, in an enterprise system with multiple data sources this should be a standalone module.

### Data Transformer

Includes data wrangling and feature engineering. For this project this module keeps the third feature in the dataset and reshapes the data. In a bigger project there would be seperate modeuls for data cleaning, data transformation and feature selection. 

### Data Preprocessing

Includes preprocessing steps required for model training. For this project this only splitting the data into training and test sets. But it could include feature scaling, one hot encoding or other required steps.

### Model Training

We use a Linear Regression model in this project. The user can choose the `learning rate`, `number of iterations` and `training mode`. For the `training mode` we have the option of `GD` (Gradient Descent) or `SGD` (Statistical Gradient Descent)
No hyper parameter optimization or cross validation is used, but in an enterprise pipeline we should include those steps.

The trained model is saved as a pickle file. Model object's `__predict` method can be called to make predictions.

### Model Evaluation

Evaluates model performance.

### API

The trained model is served as a Flask API. To create the API endpoint locally the user can run `main.py`
The API is also accible publically on "https://ksarafrazi.pythonanywhere.com". The API endpoints include:

"/model-check" :
Check model helath and information

"/stream" :
Returns prediction for a single input.

Sample API call
```
X = [0.06169621]
headers={'Content-type':'application/json'}
payload = {'data':json.dumps(X)}
y_predict = requests.post('https://ksarafrazi.pythonanywhere.com/stream', json=payload, headers=headers).json()
```

"/batch" :
Returns prediction for an array of inputs.

Sample API call
```
X = [[0.0616962065186885], [-0.0514740612388061], [0.0444512133365941]]
headers={'Content-type':'application/json'}
payload = {'data':json.dumps(X)}
y_predict = requests.post('https://ksarafrazi.pythonanywhere.com/batch', json=payload, headers=headers).json()
```




You will have to fill in the gaps in the `SimpleLinearRegression` class so that the code will run successfully.
   
The following functions need to be filled:

-  `__loss`: This function defines the loss function of your choice.
-  `__sgd`: We will use the Stochastic Gradient Descent Algorithm to optimise the slope and the intercept of our linear function. There are many resources online about SGD, However
the most important formulas are :
    

Where `n`is the number of sample in the training dataset. 

Do your best to vectorize the formulas.

-  `__predict`our linear function to predict the outcome. The function of a simple line is defined as `y= wX + b`

We have provided the benchmark code `benchmark.py`. Execute it and you should get the Coefficient of determination around `0.42`.
A good implementation should return about the same Coefficient of determination or slightly higher. During the interview we could explore the time and memory complexity of your code. 

**PS: If you are struggling implementing the above, consider using scikit-learn to progress to the next stages (but this is not encouraged).**

3. Update `main.py` to make it an API for inference. Make the API invokable from a http request. The choice of web framework is up to you. 

The API should have two endpoints:
- `POST /stream` : which takes a payload of one record and return the prediction for that record.
- `POST /batch` : which takes an array of multiple records and return an array of predictions

Think about what other features an enterprise machine learning system would have. 

#### 4. Package your code into a python package to make it easily installable and testable for developers. 

#### 5. Package your code into a container and deploy it to a container registry of your choice.
   
#### 6. Create a CICD pipeline using the technology of your choice to deploy your code to production. 
Think about what stages might be required in a full CICD pipeline. Your code should be invokable from a public URL.

#### 7. Document what componenets an enterprise machine learning system would have if you had the time to add it. 
What are some things that are critical to have versus nice to have?


## Assessment Criterion

We are not looking for a highly performant model. The criterion for this exercise is centered on a complete system that works well together and your ability to apply a machine learning inference to a real world use case. The following diagram speaks volumes about the reality of a machine learning engineer.

![img.png](mlsys.png)

We are more interested in how your overall system works and the ancillary systems and components that are considered and better yet, implemented. As you complete the challenge, try to think about the following assessment criterion:

- Does your solution work end to end?
- Are there any unit tests or integration tests?
- Has security/monitoring been considered? 
- How is your solution documented? Is it easy to build on and for other developers to understand
- How performant is your solution both from a code perspective and a scalability perspective as a service
- Has due consideration been given to what a production ML system would require? This could be interactions or dependencies with other systems.

Good luck & have fun! 

