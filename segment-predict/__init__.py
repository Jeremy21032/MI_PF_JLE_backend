import logging
import azure.functions as func
import json
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "modelo_final", "modelo_kmeans.pkl"
)
SCALER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "modelo_final", "escalador.pkl"
)
modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

DIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "modelo_final", "distancias_maximas.npy"
)
distancias_maximas = np.load(DIST_PATH)

segmentos = {
    0: "Adultos mayores con alto saldo y gasto medio",
    1: "Clientes maduros con saldo medio y bajo gasto",
    2: "Adultos jóvenes de saldo medio-bajo",
    3: "Clientes premium con altísimo saldo y gasto",
}


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    try:
        data = req.get_json()
        edad = float(data["edad"])
        genero = float(data["genero"])
        monto_promedio = float(data["monto_promedio"])
        saldo_promedio = float(data["saldo_promedio"])

        # Orden correcto: Edad, Monto_Promedio, Saldo_Promedio, Genero
        X = np.array([[edad, monto_promedio, saldo_promedio, genero]])
        X_scaled = scaler.transform(X)
        cluster = int(modelo.predict(X_scaled)[0])
        descripcion = segmentos.get(cluster, "Desconocido")
        # Calcular confianza
        centroide = modelo.cluster_centers_[cluster]
        distancia = np.linalg.norm(X_scaled - centroide)
        distancia_max = distancias_maximas[cluster]
        confianza = 1 - (distancia / distancia_max) if distancia_max > 0 else 1.0
        confianza = max(0, min(confianza, 1)) * 100  # porcentaje

        return func.HttpResponse(
            json.dumps(
                {"cluster": cluster, "descripcion": descripcion, "confianza": confianza}
            ),
            mimetype="application/json",
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}), status_code=400, mimetype="application/json"
        )
