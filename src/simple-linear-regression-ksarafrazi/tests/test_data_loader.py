import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_data
def test_data_loader():
    
    X, y = load_data()
    
    assert X.shape == (442, 10)
    assert y.shape == (442,)
    