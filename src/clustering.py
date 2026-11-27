# -*- coding: utf-8 -*-
"""
Módulo 4: Clustering de Vulnerabilidad Agrícola Provincial (No Supervisado).
Agrupa las 8 provincias de Piura usando K-Means sobre pérdidas históricas,
lluvia máxima, déficit de drenaje y porcentaje de superficie inundable.
"""

import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'provincias_piura_riesgo.csv')
MODEL_KEYS = ['scaler', 'kmeans', 'pca', 'risk_map', 'color_map', 'features']

class VulnerabilityClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.features = [
            'lluvia_max_historica_mm',
            'pct_area_inundable',
            'perdida_cosecha_nino_pct',
            'deficit_drenaje_pct'
        ]
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=2, random_state=42)

    def train_clustering(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = pd.read_csv(DATA_PATH)
        
        X = df[self.features]
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans.fit(X_scaled)
        cluster_labels = self.kmeans.labels_
        
        silhouette = silhouette_score(X_scaled, cluster_labels)
        pca_coords = self.pca.fit_transform(X_scaled)
        
        df['cluster_id'] = cluster_labels
        df['pca_x'] = pca_coords[:, 0]
        df['pca_y'] = pca_coords[:, 1]
        
        # Mapeo interpretativo del cluster basado en la pérdida promedio
        cluster_summary = df.groupby('cluster_id')['perdida_cosecha_nino_pct'].mean()
        sorted_clusters = cluster_summary.sort_values(ascending=False).index.tolist()
        
        risk_map = {
            sorted_clusters[0]: 'Vulnerabilidad Extrema (Prioridad 1)',
            sorted_clusters[1]: 'Vulnerabilidad Alta (Prioridad 2)',
            sorted_clusters[2]: 'Vulnerabilidad Moderada (Prioridad 3)'
        }
        
        color_map = {
            sorted_clusters[0]: '#d9534f', # Rojo
            sorted_clusters[1]: '#f0ad4e', # Naranja
            sorted_clusters[2]: '#5cb85c'  # Verde
        }
        
        df['nivel_vulnerabilidad'] = df['cluster_id'].map(risk_map)
        df['color_alerta'] = df['cluster_id'].map(color_map)
        
        self.risk_map, self.color_map = risk_map, color_map

        # Guardar artefactos (formato único que espera load_model)
        joblib.dump({
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'pca': self.pca,
            'risk_map': {int(k): v for k, v in risk_map.items()},
            'color_map': {int(k): v for k, v in color_map.items()},
            'features': self.features,
            'silhouette': float(silhouette),
            'sklearn_version': sklearn.__version__
        }, os.path.join(MODEL_DIR, 'kmeans_provincias.pkl'))
        
        # Guardar resultados
        out_csv = os.path.join(os.path.dirname(DATA_PATH), 'provincias_piura_clustered.csv')
        df.to_csv(out_csv, index=False, encoding='utf-8')
        
        print(f"[OK] Clustering K-Means completado. Coeficiente Silhouette: {round(silhouette, 3)}")
        return df, round(silhouette, 3)

    def load_model(self):
        """Carga models/kmeans_provincias.pkl (scaler + kmeans + pca + mapas de riesgo).
        Si falta o le faltan claves, reentrena y vuelve a exportar."""
        model_file = os.path.join(MODEL_DIR, 'kmeans_provincias.pkl')
        try:
            pack = joblib.load(model_file)
            faltantes = set(MODEL_KEYS) - set(pack)
            if faltantes:
                raise ValueError(f'faltan claves {sorted(faltantes)}')
        except Exception as e:
            print(f"[AVISO] No se pudo cargar {model_file} ({e}). Reentrenando...")
            self.train_clustering()
            pack = joblib.load(model_file)
        self.scaler, self.kmeans, self.pca = pack['scaler'], pack['kmeans'], pack['pca']
        self.features = pack['features']
        self.risk_map, self.color_map = pack['risk_map'], pack['color_map']
        return pack

    def assign_clusters(self, df):
        """Aplica el modelo exportado a un DataFrame con las 4 variables de riesgo."""
        if not hasattr(self, 'risk_map'):
            self.load_model()
        df = df.copy()
        X_scaled = self.scaler.transform(df[self.features])
        df['cluster_id'] = self.kmeans.predict(X_scaled)
        coords = self.pca.transform(X_scaled)
        df['pca_x'], df['pca_y'] = coords[:, 0], coords[:, 1]
        df['nivel_vulnerabilidad'] = df['cluster_id'].map(self.risk_map)
        df['color_alerta'] = df['cluster_id'].map(self.color_map)
        return df

    def get_clustered_data(self):
        """Zonificación de las 8 provincias calculada con el modelo exportado en models/."""
        return self.assign_clusters(pd.read_csv(DATA_PATH))

if __name__ == '__main__':
    clusterer = VulnerabilityClusterer()
    df_res, sil = clusterer.train_clustering()
    print(df_res[['provincia', 'cultivo_predominante', 'nivel_vulnerabilidad', 'perdida_cosecha_nino_pct']])
