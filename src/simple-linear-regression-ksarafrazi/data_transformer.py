import numpy as np

def feature_selection(X):
    """
    Returns only the third feature present in the data
    Args:
        X: input data
    Returns:
        X_transformed: extracted features
    """
    
    # Use only one feature
    X_transformed = X[:, np.newaxis, 2]
    
    return X_transformed

def train_test_split(X,y,n):
    """
    Splits data into training and test sets by keeping the the last n samples as test, and using the rest for training.
    Args:
        X: input data
        y: target data
        n: number of test samples
    Returns:
        X_train: training input data
        X_test: testing input data
        y_train: training labels
        y_test: testing labels
    """
    
    assert len(X)==len(y), "Input data and labels must have the same length"
    assert n < len(X), "Number of test samples can not be more than the total number of samples."
    
     # Split the data into training/testing sets
    X_train = X[:-n]
    X_test = X[-n:]

    # Split the targets into training/testing sets
    y_train = y[:-n].reshape(-1,1)
    y_test = y[-n:].reshape(-1,1)
    
    return X_train,X_test,y_train,y_test
    
    