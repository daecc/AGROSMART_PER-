# -*- coding: utf-8 -*-
"""
Módulo 1: Predicción Supervisada de Rendimiento Agrícola ante El Niño.
Usa Random Forest Regressor para predecir el rendimiento (t/ha) y el porcentaje
de pérdida frente al promedio histórico según variables climáticas (ICEN, TSM, lluvias).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tabla_maestra_agro_nino.csv')

class CropYieldPredictor:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.features = ['icen', 'anomalia_tsm', 'lluvia_mensual_mm', 'temp_max_prom', 'mes']
        self.target = 'rendimiento_tha'

    def train_models(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = pd.read_csv(DATA_PATH)
        
        for cultivo in ['Arroz', 'Limon']:
            df_c = df[df['cultivo'] == cultivo].copy()
            X = df_c[self.features]
            y = df_c[self.target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.models[cultivo] = rf
            self.metrics[cultivo] = {
                'r2': round(r2, 4),
                'mae': round(mae, 3),
                'rmse': round(rmse, 3),
                'baseline_promedio': round(df_c[self.target].mean(), 2)
            }
            
            model_file = os.path.join(MODEL_DIR, f'rf_yield_{cultivo.lower()}.pkl')
            joblib.dump(rf, model_file)
            
        print("[OK] Modelos de predicción de rendimiento entrenados con éxito.")
        return self.metrics

    def load_models(self):
        for cultivo in ['Arroz', 'Limon']:
            model_file = os.path.join(MODEL_DIR, f'rf_yield_{cultivo.lower()}.pkl')
            if os.path.exists(model_file):
                self.models[cultivo] = joblib.load(model_file)
            else:
                self.train_models()
                break

    def predict(self, cultivo, icen, anomalia_tsm, lluvia_mm, temp_max, mes=3):
        if cultivo not in self.models:
            self.load_models()
            
        model = self.models.get(cultivo)
        if not model:
            self.train_models()
            model = self.models.get(cultivo)

        input_data = pd.DataFrame([{
            'icen': float(icen),
            'anomalia_tsm': float(anomalia_tsm),
            'lluvia_mensual_mm': float(lluvia_mm),
            'temp_max_prom': float(temp_max),
            'mes': int(mes)
        }])

        rendimiento_pred = model.predict(input_data)[0]
        
        # Rendimientos de referencia en condiciones normales en Piura
        base_normal = 8.8 if cultivo == 'Arroz' else 12.5
        perdida_pct = max(0.0, ((base_normal - rendimiento_pred) / base_normal) * 100.0)
        
        # Clasificación de riesgo agronómico
        if perdida_pct < 15:
            alerta = "Baja / Normal"
            color = "green"
        elif perdida_pct < 35:
            alerta = "Moderada / Alerta Amarilla"
            color = "orange"
        else:
            alerta = "Crítica / Alerta Roja (Emergencia Fitosanitaria)"
            color = "red"

        return {
            'cultivo': cultivo,
            'rendimiento_estimado_tha': round(rendimiento_pred, 2),
            'rendimiento_historico_base': base_normal,
            'perdida_estimada_pct': round(perdida_pct, 1),
            'alerta_riesgo': alerta,
            'color': color,
            'importancia_variables': dict(zip(self.features, [round(f, 3) for f in model.feature_importances_]))
        }

if __name__ == '__main__':
    predictor = CropYieldPredictor()
    metrics = predictor.train_models()
    print("Métricas:", metrics)
    test_pred = predictor.predict('Limon', icen=2.4, anomalia_tsm=2.7, lluvia_mm=380, temp_max=35.0, mes=3)
    print("Predicción prueba Limón en Niño Fuerte:", test_pred)
