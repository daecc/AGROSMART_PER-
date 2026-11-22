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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'provincias_piura_riesgo.csv')

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
        
        # Guardar artefactos
        joblib.dump({
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'pca': self.pca,
            'risk_map': risk_map,
            'color_map': color_map,
            'features': self.features
        }, os.path.join(MODEL_DIR, 'kmeans_provincias.pkl'))
        
        # Guardar resultados
        out_csv = os.path.join(os.path.dirname(DATA_PATH), 'provincias_piura_clustered.csv')
        df.to_csv(out_csv, index=False, encoding='utf-8')
        
        print(f"[OK] Clustering K-Means completado. Coeficiente Silhouette: {round(silhouette, 3)}")
        return df, round(silhouette, 3)

    def get_clustered_data(self):
        out_csv = os.path.join(os.path.dirname(DATA_PATH), 'provincias_piura_clustered.csv')
        if not os.path.exists(out_csv):
            df, _ = self.train_clustering()
            return df
        return pd.read_csv(out_csv)

if __name__ == '__main__':
    clusterer = VulnerabilityClusterer()
    df_res, sil = clusterer.train_clustering()
    print(df_res[['provincia', 'cultivo_predominante', 'nivel_vulnerabilidad', 'perdida_cosecha_nino_pct']])
