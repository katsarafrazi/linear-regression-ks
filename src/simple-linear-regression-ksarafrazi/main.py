from flask import Flask, jsonify, make_response
import pickle
import numpy as np
import json
from simple_linear_regr import SimpleLinearRegression
from flask_restful import reqparse
import os

app = Flask(__name__)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Endpoint for streaming mode
@app.route("/stream/", methods =["POST"])
def stream():
    
    args = parser.parse_args()
    X = np.array(json.loads(args['data']))
    prediction = model.predict(X)
    
    return {
        'Input': X[0],
        'Prediction': prediction[0]
}

# Endpoint for batch mode
@app.route("/batch/", methods =["POST"])
def batch():
    
    args = parser.parse_args()
    X = np.array(json.loads(args['data']))
    prediction = model.predict(X)
    
    return {
        'Input': X.tolist(),
        'Prediction': prediction[:,0].tolist()
}

@app.route("/model-check", methods =["GET"])
def model_check():  
    return {
        'Model Type': 'SimpleLinearRegression', 
        'Version':model.version, 
        'Number of Inputs':len(model.W[0]), 
        'Weights': str(model.W[0]), 
        'Bias': str(model.b[0]), 
        'Loss':str(model.losses[-1]),
        'r2 Coef':str(model.r2)
}

if __name__ == '__main__':
    # Load model
    with open(os.path.join(os.getcwd(),'model','model.pickle'), 'rb') as f:
        model = pickle.load(f)

    app.run(debug=True)