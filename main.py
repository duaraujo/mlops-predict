# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

# nao foi usado no treino
# 'QuoteNumber', 'Field7', 'CoverageField6A', 

colunas = ['CoverageField6B', 'CoverageField8',
            'CoverageField9', 'CoverageField11B', 'SalesField1A', 'SalesField1B', 'SalesField3',
            'SalesField4', 'SalesField5', 'PersonalField1', 'PersonalField2', 'PersonalField9',
            'PersonalField10A', 'PersonalField10B', 'PersonalField12', 'PropertyField34', 
            'PropertyField35',  'PropertyField37']

app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = 'BASIC_AUTH_USERNAME'
app.config["BASIC_AUTH_PASSWORD"] = 'BASIC_AUTH_PASSWORD'

basic_auth = BasicAuth(app)

def load_model(file_name = 'model_xgboost.pkl'):
    return pickle.load(open(file_name, "rb"))

modelo = load_model(file_name = 'models/model_xgboost.pkl')

@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    # Pegar o JSON da requisição
    dados = request.get_json()
    payload = [dados[col] for col in colunas]
    payload = xgb.DMatrix([payload], feature_names=colunas)
    result = np.float64(modelo.predict(payload)[0])

    return jsonify(resultado = result)

# Rota padrão
@app.route('/')
def home():
    print("Executou rota inicial")
    return 'API de Predição de potenciais clientes'

# Subir a API
if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0')
