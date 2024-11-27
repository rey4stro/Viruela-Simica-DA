from flask import Blueprint, render_template, request
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

def predecir_calidad(Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity):
    # Cargar el scaler ajustado
    scaler = joblib.load(skalater_path)

    # Crear el array de entrada
    entrada = np.array([[Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity]])

    # Normalizar la entrada
    entrada_normalizada = scaler.transform(entrada)

    # Realizar la predicción con el modelo entrenado
    prediccion = loaded_model.predict(entrada_normalizada)

    # Interpretar la predicción
    if prediccion[0] >= 0.5:
        return "La calidad de la fruta es buena"
    else:
        return "La calidad de la fruta es mala"








@main.route('/')
def index():
    return render_template('index.html')


@main.route('/submit', methods=['POST'])
def guardar():

    size = request.form.get('size')
    weight = request.form.get('weight')
    sweetness = request.form.get('sweetness')
    softness = request.form.get('softness')
    harvest_time = request.form.get('harvest_time')
    ripeness = request.form.get('ripeness')
    acidity = request.form.get('acidity')

    resultado = predecir_calidad(
    Size=size, Weight=weight, Sweetness=sweetness,
    Softness=softness, HarvestTime=harvest_time, Ripeness=ripeness, Acidity=acidity)

    

    return f"Resultado: {resultado}"

