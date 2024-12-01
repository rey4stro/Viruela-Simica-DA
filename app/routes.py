from flask import Blueprint, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib




save_path = os.path.join('app', 'static', 'banana_quality.h5')
skalater_path = os.path.join('app', 'static', 'scaler.pkl')

# Cargar el modelo
loaded_model = load_model(save_path)
print("Modelo cargado correctamente.")

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/submit', methods=['POST'])
def guardar():

    Size = request.form.get('size')
    Weight = request.form.get('weight')
    Sweetness = request.form.get('sweetness')
    Softness = request.form.get('softness')
    HarvestTime = request.form.get('harvest_time')
    Ripeness = request.form.get('ripeness')
    Acidity = request.form.get('acidity')

   
    scaler = joblib.load(skalater_path)

    # Crear el array de entrada
    entrada = np.array([[Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity]])

    # Normalizar la entrada
    entrada_normalizada = scaler.transform(entrada)

    # Realizar la predicción con el modelo entrenado
    prediccion = loaded_model.predict(entrada_normalizada)

    probabilidad = float(prediccion[0][0])
    resultado = ""

    
    if prediccion[0] >= 0.5:
        resultado = "La calidad de la fruta es buena"
        imagen = url_for('static', filename='buena.png')  # Ruta de imagen buena
    else:
        resultado = "La calidad de la fruta es mala"
        imagen = url_for('static', filename='mala.png')  # Ruta de imagen buena


    # Devolver el resultado como una respuesta JSON
    return jsonify({'prediction': probabilidad, 'result': resultado, 'imagen': imagen})



