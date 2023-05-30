from flask import Flask
import numpy as np
import json
from flask_restful import reqparse
import os

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

def load_model():
    # Load model
    with open('model/model_json.json', 'r') as openfile:

    # Reading from json file
        model = json.load(openfile)
    return model

model = load_model()
app = Flask(__name__)

# Endpoint for streaming mode
@app.route("/stream/", methods =["POST"])
def stream():

    args = parser.parse_args()
    X = np.array(json.loads(args['data']))
    prediction = float(model['Weights'][1:-1]) * X + float(model['Bias'])

    return {
        'Input': X[0],
        'Prediction': prediction[0]
}

# Endpoint for batch mode
@app.route("/batch/", methods =["POST"])
def batch():

    args = parser.parse_args()
    X = np.array(json.loads(args['data']))
    prediction = float(model['Weights'][1:-1]) * X + float(model['Bias'])

    return {
        'Input': X.tolist(),
        'Prediction': prediction.tolist()
}

@app.route("/model-check", methods =["GET"])
def model_check():
    return model

@app.route("/", methods =["GET"])
def welcome():
    return "Welcome to Linear Regression API."

if __name__ == '__main__':
    app.run(debug=True)