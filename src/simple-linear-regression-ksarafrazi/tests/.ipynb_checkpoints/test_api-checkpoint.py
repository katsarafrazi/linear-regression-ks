import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
import json

HEADERS={
    'Content-type':'application/json'
}
API_PATH = 'http://127.0.0.1:5000'

def test_stream_mode():    
    X = [0.06169621]

    payload = {'data':json.dumps(X)}
    res = requests.post(API_PATH + '/stream', json=payload, headers=HEADERS)
    assert res.status_code == 200
    
    y_predict = res.json()
    
    assert not np.isnan(y_predict['Prediction'])
    
def test_batch_mode():    
    X = [[0.0616962065186885], [-0.0514740612388061], [0.0444512133365941], [-0.0115950145052127], [-0.0363846922044735], [-0.0406959404999971], [-0.0471628129432825], [-0.00189470584028465], [0.0616962065186885]]

    payload = {'data':json.dumps(X)}
    res = requests.post(API_PATH + '/batch', json=payload, headers=HEADERS)
    assert res.status_code == 200

    y_predict = res.json()

    assert len(y_predict['Input']) == len(y_predict['Prediction'])
    assert sum(np.isnan(y_predict['Prediction'])) == 0
    