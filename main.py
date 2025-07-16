
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

colunas = ['tamaho','ano', 'garagem']
modelo = pickle.load(open('modelo.sav','rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'gustavo'
app.config['BASIC_AUTH_PASSWORD'] = 'alura'


basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API com Flask"
#isso so e executado quando alguem entra no endpoint 
@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
        tb = TextBlob(frase)
        polaridade = tb.sentiment.polarity
        return f"Frase: {frase} | Polaridade: {polaridade}"

@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True)
