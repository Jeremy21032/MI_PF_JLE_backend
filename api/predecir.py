import json
import joblib
import numpy as np
import os

def handler(request):
    try:
        # Cargar modelo y escalador solo una vez (cache en el entorno serverless)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "modelo_final", "modelo_kmeans.pkl")
        SCALER_PATH = os.path.join(BASE_DIR, "modelo_final", "escalador.pkl")
        DIST_PATH = os.path.join(BASE_DIR, "modelo_final", "distancias_maximas.npy")

        if not hasattr(handler, "_modelo"):
            handler._modelo = joblib.load(MODEL_PATH)
        if not hasattr(handler, "_scaler"):
            handler._scaler = joblib.load(SCALER_PATH)
        if not hasattr(handler, "_distancias_maximas"):
            handler._distancias_maximas = np.load(DIST_PATH)

        modelo = handler._modelo
        scaler = handler._scaler
        distancias_maximas = handler._distancias_maximas

        segmentos = {
            0: "Adultos mayores con alto saldo y gasto medio",
            1: "Clientes maduros con saldo medio y bajo gasto",
            2: "Adultos jóvenes de saldo medio-bajo",
            3: "Clientes premium con altísimo saldo y gasto",
            4: "Jóvenes con bajo saldo y bajo consumo"
        }

        data = request.json()
        edad = float(data["edad"])
        genero = float(data["genero"])
        monto_promedio = float(data["monto_promedio"])
        saldo_promedio = float(data["saldo_promedio"])

        X = np.array([[edad, monto_promedio, saldo_promedio, genero]])
        X_scaled = scaler.transform(X)
        cluster = int(modelo.predict(X_scaled)[0])
        descripcion = segmentos.get(cluster, "Desconocido")

        centroide = modelo.cluster_centers_[cluster]
        distancia = np.linalg.norm(X_scaled - centroide)
        distancia_max = distancias_maximas[cluster]
        confianza = 1 - (distancia / distancia_max) if distancia_max > 0 else 1.0
        confianza = max(0, min(confianza, 1)) * 100

        return {
            "statusCode": 200,
            "body": json.dumps({
                "cluster": cluster,
                "descripcion": descripcion,
                "confianza": confianza
            }),
            "headers": {
                "Content-Type": "application/json"
            }
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json"
            }
        } 