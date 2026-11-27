# -*- coding: utf-8 -*-
"""
Módulo 2: Series Temporales con Variable Exógena (ICEN de El Niño).
Pronostica la evolución mensual del precio en chacra (S/ por kg) y producción
en Piura a 6 y 12 meses vista, modelando la respuesta al calentamiento del mar.
"""

import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tabla_maestra_agro_nino.csv')

# Variables que forecast_future() sabe construir en cada paso del pronóstico
EXPECTED_FEATURES = [
    'precio_chacra_soles_kg_lag1', 'precio_chacra_soles_kg_lag2', 'precio_chacra_soles_kg_lag3',
    'icen_lag1', 'icen_lag2', 'icen_lag3', 'lluvia_lag1', 'lluvia_lag2', 'lluvia_lag3',
    'mes_sin', 'mes_cos', 'icen', 'anomalia_tsm'
]

class AgroTimeSeriesForecaster:
    def __init__(self):
        self.models = {}
        self.lags = 3

    def prepare_lagged_features(self, df_sub, target_col='precio_chacra_soles_kg'):
        df = df_sub.sort_values('fecha').copy()
        
        # Generar retardos (lags) del precio y del ICEN
        for i in range(1, self.lags + 1):
            df[f'{target_col}_lag{i}'] = df[target_col].shift(i)
            df[f'icen_lag{i}'] = df['icen'].shift(i)
            df[f'lluvia_lag{i}'] = df['lluvia_mensual_mm'].shift(i)

        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df = df.dropna()
        
        feature_cols = [c for c in df.columns if 'lag' in c or 'mes_' in c or c in ['icen', 'anomalia_tsm']]
        return df, feature_cols

    def train_models(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df_all = pd.read_csv(DATA_PATH)
        metrics = {}

        for cultivo in ['Arroz', 'Limon']:
            df_c = df_all[df_all['cultivo'] == cultivo].copy()
            df_lag, feat_cols = self.prepare_lagged_features(df_c, target_col='precio_chacra_soles_kg')
            
            train_size = int(len(df_lag) * 0.85)
            train_df = df_lag.iloc[:train_size]
            test_df = df_lag.iloc[train_size:]
            
            X_train = train_df[feat_cols]
            y_train = train_df['precio_chacra_soles_kg']
            X_test = test_df[feat_cols]
            y_test = test_df['precio_chacra_soles_kg']
            
            model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.models[cultivo] = {
                'model': model,
                'features': feat_cols,
                'last_rows': df_c.tail(self.lags + 2)
            }
            
            metrics[cultivo] = {
                'mape_pct': round(float(mape), 2),
                'rmse_soles': round(float(rmse), 2)
            }
            
            self.models[cultivo]['metrics'] = metrics[cultivo]
            self.models[cultivo]['sklearn_version'] = sklearn.__version__
            out_file = os.path.join(MODEL_DIR, f'ts_precio_{cultivo.lower()}.pkl')
            joblib.dump(self.models[cultivo], out_file)

        print("[OK] Modelos de series temporales entrenados con éxito.")
        return metrics

    def load_model(self, cultivo):
        """Carga models/ts_precio_<cultivo>.pkl. Si falta, está corrupto o no tiene las
        variables que espera forecast_future(), reentrena y vuelve a exportar."""
        if cultivo in self.models:
            return self.models[cultivo]
        model_file = os.path.join(MODEL_DIR, f'ts_precio_{cultivo.lower()}.pkl')
        try:
            pack = joblib.load(model_file)
            faltantes = set(pack['features']) - set(EXPECTED_FEATURES)
            if faltantes or not hasattr(pack['model'], 'predict'):
                raise ValueError(f'variables incompatibles: {sorted(faltantes)}')
            self.models[cultivo] = pack
        except Exception as e:
            print(f"[AVISO] No se pudo cargar {model_file} ({e}). Reentrenando...")
            self.train_models()
        return self.models[cultivo]

    def forecast_future(self, cultivo='Limon', months_ahead=6, escenario_nino="Fuerte (ICEN +2.2°C)"):
        saved_pack = self.load_model(cultivo)
        model = saved_pack['model']
        feat_cols = saved_pack['features']
        
        # Definir el valor de ICEN según el escenario elegido para el pronóstico
        if "Extraordinario" in escenario_nino:
            future_icen = 2.6
            future_anomalia = 2.9
            future_lluvia = 300.0
        elif "Fuerte" in escenario_nino:
            future_icen = 2.1
            future_anomalia = 2.3
            future_lluvia = 220.0
        elif "Moderado" in escenario_nino:
            future_icen = 1.4
            future_anomalia = 1.5
            future_lluvia = 80.0
        else: # Normal / Neutro
            future_icen = 0.2
            future_anomalia = 0.1
            future_lluvia = 15.0

        # Cargar historial reciente
        df_all = pd.read_csv(DATA_PATH)
        df_c = df_all[df_all['cultivo'] == cultivo].sort_values('fecha').copy()
        
        # Simular proyección paso a paso (autoregresiva)
        history_prices = list(df_c['precio_chacra_soles_kg'].values[-self.lags:])
        history_icen = list(df_c['icen'].values[-self.lags:])
        history_lluvia = list(df_c['lluvia_mensual_mm'].values[-self.lags:])
        
        last_date = pd.to_datetime(df_c['fecha'].max())
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]
        forecasted_prices = []
        
        for dt in future_dates:
            month = dt.month
            row_dict = {
                'precio_chacra_soles_kg_lag1': history_prices[-1],
                'precio_chacra_soles_kg_lag2': history_prices[-2],
                'precio_chacra_soles_kg_lag3': history_prices[-3],
                'icen_lag1': history_icen[-1],
                'icen_lag2': history_icen[-2],
                'icen_lag3': history_icen[-3],
                'lluvia_lag1': history_lluvia[-1],
                'lluvia_lag2': history_lluvia[-2],
                'lluvia_lag3': history_lluvia[-3],
                'mes_sin': np.sin(2 * np.pi * month / 12),
                'mes_cos': np.cos(2 * np.pi * month / 12),
                'icen': future_icen,
                'anomalia_tsm': future_anomalia
            }
            
            X_step = pd.DataFrame([row_dict])[feat_cols]
            pred_price = max(1.0, float(model.predict(X_step)[0]))
            forecasted_prices.append(round(pred_price, 2))
            
            # Actualizar historia para el siguiente paso
            history_prices.append(pred_price)
            history_icen.append(future_icen)
            history_lluvia.append(future_lluvia)

        df_future = pd.DataFrame({
            'fecha': [d.strftime('%Y-%m') for d in future_dates],
            'precio_pronosticado_soles_kg': forecasted_prices,
            'escenario_climatico': escenario_nino
        })
        
        # Historial reciente para graficar (últimos 18 meses)
        df_hist = df_c.tail(18)[['fecha', 'precio_chacra_soles_kg', 'icen']].copy()
        df_hist['fecha'] = pd.to_datetime(df_hist['fecha']).dt.strftime('%Y-%m')
        
        return df_hist, df_future

if __name__ == '__main__':
    ts = AgroTimeSeriesForecaster()
    m = ts.train_models()
    print("Métricas Series:", m)
    hist, fut = ts.forecast_future('Limon', months_ahead=6, escenario_nino="Fuerte (ICEN +2.2°C)")
    print("Pronóstico Limón:", fut)
