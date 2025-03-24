import pickle
import os
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import Model, IntegerField, FloatField, TextField, IntegrityError
from playhouse.db_url import connect
from playhouse.shortcuts import model_to_dict

# Connection to SQLite database (or PostgreSQL if DATABASE_URL is configured)
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

# Prediction model
class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

# Create the table if it does not exist
DB.create_tables([Prediction], safe=True)

# Load the previously trained model and other information
with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# Initialize Flask
app = Flask(__name__)

# /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the JSON request
    obs_dict = request.get_json()

    # Validate required fields
    required_fields = ['id', 'observation']
    for field in required_fields:
        if field not in obs_dict:
            return jsonify({'error': f'{field} is required'}), 400

    # Get the observation data
    _id = obs_dict['id']
    observation = obs_dict['observation']
    
    # Validate observation fields
    required_observation_fields = ['age', 'education', 'hours-per-week', 'native-country']
    for field in required_observation_fields:
        if field not in observation:
            return jsonify({'error': f'{field} is required in observation'}), 400

    # Convert the observation into a DataFrame
    try:
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    except ValueError as e:
        return jsonify({'error': f'Invalid value in observation: {e}'}), 400

    # Get the probability of the positive class
    proba = pipeline.predict_proba(obs)[0, 1]
    
    # Check if the ID already exists in the database
    if Prediction.select().where(Prediction.observation_id == _id).exists():
        existing_pred = Prediction.get(Prediction.observation_id == _id)
        response = {
            'error': f'Observation ID {_id} already exists',
            'proba': existing_pred.proba
        }
        return jsonify(response), 400

    # Create a new entry in the database
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=json.dumps(observation)  
    )
    try:
        p.save()  
    except IntegrityError:
        error_msg = f'Observation ID {_id} already exists'
        response = {'error': error_msg}
        return jsonify(response), 400

    # Return the response with the probability
    response = {'proba': proba}
    return jsonify(response)

# /update endpoint
@app.route('/update', methods=['POST'])
def update_true_class():
    # Get data from the JSON request
    data = request.get_json()
    
    # Check if the necessary fields are present
    if 'id' not in data or 'true_class' not in data:
        return jsonify({'error': 'Both id and true_class are required'}), 400
    
    _id = data['id']
    true_class = data['true_class']
    
    # Check if the observation exists in the database
    try:
        prediction = Prediction.get(Prediction.observation_id == _id)
    except Prediction.DoesNotExist:
        return jsonify({'error': f'Observation with id {_id} not found'}), 404
    
    # Update the true_class field of the observation
    prediction.true_class = true_class
    prediction.save()
    
    # Return the updated observation
    response = {
        'id': prediction.observation_id,
        'true_class': prediction.true_class,
        'observation': json.loads(prediction.observation)
    }
    
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5007)
