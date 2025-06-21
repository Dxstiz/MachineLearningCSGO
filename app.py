# Contenido para tu nuevo archivo app.py

from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo desde la carpeta checkpoints
model = joblib.load('checkpoints/clasificador_roundwinner_csgo.joblib')

@app.route('/')
def home():
    # Esta ruta muestra nuestro formulario principal
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Recolectar todos los datos del formulario
    form_data = request.form.to_dict()

    # Rellenar datos faltantes con valores por defecto para completar las 16 features
    # Es importante mantener el orden y los nombres de las columnas
    features_list = [
        'Map', 'Team', 'TravelledDistance', 'RLethalGrenadesThrown',
        'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle',
        'PrimarySMG', 'PrimaryHeavy', 'PrimaryPistol', 'RoundKills',
        'RoundAssists', 'RoundFlankKills', 'RoundHeadshots',
        'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue'
    ]
    
    # Creamos un diccionario completo con los datos del form y los placeholders
    data_for_prediction = {
        'Map': form_data.get('Map', 'de_dust2'),
        'Team': form_data.get('Team', 'Counter-Terrorist'),
        'RoundKills': int(form_data.get('RoundKills', 0)),
        'TravelledDistance': 2000, 'RLethalGrenadesThrown': 1, 'RNonLethalGrenadesThrown': 1,
        'PrimaryAssaultRifle': 0.8, 'PrimarySniperRifle': 0.0, 'PrimarySMG': 0.1,
        'PrimaryHeavy': 0.0, 'PrimaryPistol': 0.1, 'RoundAssists': 0, 'RoundFlankKills': 0,
        'RoundHeadshots': 0, 'RoundStartingEquipmentValue': 3000, 'TeamStartingEquipmentValue': 15000
    }
    
    # Convertimos a DataFrame
    input_df = pd.DataFrame([data_for_prediction], columns=features_list)

    # Realizar la predicción
    prediction = model.predict(input_df)
    
    # Preparar el texto del resultado
    if prediction[0] == 1:
        output_text = "El equipo probablemente GANA la ronda."
    else:
        output_text = "El equipo probablemente PIERDE la ronda."

    # Devolvemos el resultado renderizando la plantilla result.html
    return render_template('result.html', prediction_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)