from sklearn.datasets import load_diabetes

def load_data():
    """
    Load input data
    """
    
    # Load the diabetes dataset
    X, y = load_diabetes(return_X_y=True)
    
    return X,y