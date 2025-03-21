import pickle
import os
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import Model, IntegerField, FloatField, TextField, IntegrityError
from playhouse.db_url import connect
from playhouse.shortcuts import model_to_dict

# Conexão ao banco de dados SQLite (ou PostgreSQL se DATABASE_URL estiver configurado)
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

# Modelo de previsão
class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

# Criação da tabela, caso não exista
DB.create_tables([Prediction], safe=True)

# Carregar o modelo previamente treinado e outras informações
with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# Inicializar o Flask
app = Flask(__name__)

# Endpoint /predict
@app.route('/predict', methods=['POST'])
def predict():
    # Obter dados da requisição JSON
    obs_dict = request.get_json()

    # Validar campos obrigatórios
    required_fields = ['id', 'observation']
    for field in required_fields:
        if field not in obs_dict:
            return jsonify({'error': f'{field} is required'}), 400

    # Obter os dados da observação
    _id = obs_dict['id']
    observation = obs_dict['observation']
    
    # Validar os campos da observação
    required_observation_fields = ['age', 'education', 'hours-per-week', 'native-country']
    for field in required_observation_fields:
        if field not in observation:
            return jsonify({'error': f'{field} is required in observation'}), 400

    # Converter a observação para um DataFrame
    try:
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    except ValueError as e:
        return jsonify({'error': f'Invalid value in observation: {e}'}), 400

    # Obter a probabilidade da classe positiva
    proba = pipeline.predict_proba(obs)[0, 1]
    
    # Verificar se o ID já existe no banco de dados
    if Prediction.select().where(Prediction.observation_id == _id).exists():
        existing_pred = Prediction.get(Prediction.observation_id == _id)
        response = {
            'error': f'Observation ID {_id} already exists',
            'proba': existing_pred.proba
        }
        return jsonify(response), 400

    # Criar uma nova entrada no banco de dados
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=json.dumps(observation)  # Armazenar a observação como JSON
    )
    try:
        p.save()  # Salvar no banco
    except IntegrityError:
        error_msg = f'Observation ID {_id} already exists'
        response = {'error': error_msg}
        return jsonify(response), 400

    # Retornar a resposta com a probabilidade
    response = {'proba': proba}
    return jsonify(response)

@app.route('/update', methods=['POST'])
def update_true_class():
    # Obter dados da requisição JSON
    data = request.get_json()
    
    # Verificar se os campos necessários estão presentes
    if 'id' not in data or 'true_class' not in data:
        return jsonify({'error': 'Both id and true_class are required'}), 400
    
    _id = data['id']
    true_class = data['true_class']
    
    # Verificar se a observação existe no banco de dados
    try:
        prediction = Prediction.get(Prediction.observation_id == _id)
    except Prediction.DoesNotExist:
        return jsonify({'error': f'Observation with id {_id} not found'}), 404
    
    # Atualizar o campo true_class da observação
    prediction.true_class = true_class
    prediction.save()
    
    # Retornar a observação atualizada
    response = {
        'id': prediction.observation_id,
        'true_class': prediction.true_class,
        'observation': json.loads(prediction.observation)
    }
    
    return jsonify(response), 200


# Rodar a aplicação
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5007)
