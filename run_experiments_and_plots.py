# -*- coding: utf-8 -*-
"""
Ejecutor de experimentos y generador de gráficos de evaluación
para el Informe Técnico y Cuaderno de AgroSmart Perú.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUT_DIR = os.path.join(os.path.dirname(__file__), 'Documentos', 'graficos_evaluacion')
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv('data/tabla_maestra_agro_nino.csv')
df['fecha'] = pd.to_datetime(df['fecha'])

# 1. Matriz de Correlación
fig, ax = plt.subplots(figsize=(8, 6))
cols_corr = ['icen', 'anomalia_tsm', 'lluvia_mensual_mm', 'temp_max_prom', 'rendimiento_tha', 'precio_chacra_soles_kg']
corr_matrix = df[cols_corr].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Matriz de Correlación: Variables El Niño vs Rendimiento y Precios en Piura')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_matriz_correlacion.png'), dpi=300)
plt.close()
print("[OK] Gráfico 1: 01_matriz_correlacion.png")

# 2. Rendimiento Real vs Predicho e Importancia
df_limon = df[df['cultivo'] == 'Limon'].copy()
features = ['icen', 'anomalia_tsm', 'lluvia_mensual_mm', 'temp_max_prom', 'mes']
X = df_limon[features]
y = df_limon['rendimiento_tha']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.scatter(y_test, y_pred_rf, color='#2e7d32', alpha=0.7, edgecolors='k', label='Datos de Prueba')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ajuste Ideal')
ax1.set_xlabel('Rendimiento Real (t/ha)')
ax1.set_ylabel('Rendimiento Predicho (t/ha)')
ax1.set_title(f'Random Forest: Real vs Predicho (R² = {r2_score(y_test, y_pred_rf):.3f})')
ax1.legend()

importancias = pd.Series(rf_model.feature_importances_, index=features).sort_values()
importancias.plot(kind='barh', color='#388e3c', ax=ax2)
ax2.set_xlabel('Importancia Relativa (Gini)')
ax2.set_title('Variables Determinantes en la Caída de Rendimiento')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_evaluacion_prediccion_rendimiento.png'), dpi=300)
plt.close()
print("[OK] Gráfico 2: 02_evaluacion_prediccion_rendimiento.png")

# 3. Pronóstico de Series Temporales
df_ts = df_limon.sort_values('fecha').copy()
for i in range(1, 4):
    df_ts[f'precio_lag{i}'] = df_ts['precio_chacra_soles_kg'].shift(i)
    df_ts[f'icen_lag{i}'] = df_ts['icen'].shift(i)
df_ts['mes_sin'] = np.sin(2 * np.pi * df_ts['mes'] / 12)
df_ts['mes_cos'] = np.cos(2 * np.pi * df_ts['mes'] / 12)
df_ts = df_ts.dropna()
feat_ts = [c for c in df_ts.columns if 'lag' in c or 'mes_' in c or c in ['icen', 'anomalia_tsm']]
train_size = len(df_ts) - 36
X_train_ts, X_test_ts = df_ts[feat_ts].iloc[:train_size], df_ts[feat_ts].iloc[train_size:]
y_train_ts, y_test_ts = df_ts['precio_chacra_soles_kg'].iloc[:train_size], df_ts['precio_chacra_soles_kg'].iloc[train_size:]
fechas_test = df_ts['fecha'].iloc[train_size:]

gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
gbr.fit(X_train_ts, y_train_ts)
y_pred_ts = gbr.predict(X_test_ts)
mape = mean_absolute_percentage_error(y_test_ts, y_pred_ts) * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(fechas_test, y_test_ts.values, 'o-', color='#1565c0', label='Precio Real BCRP/MIDAGRI (S/ por kg)')
ax.plot(fechas_test, y_pred_ts, 's--', color='#d32f2f', label=f'Pronóstico Modelo Exógeno (MAPE: {mape:.2f}%)')
ax.set_ylabel('Soles por Kilogramo (S/)')
ax.set_title('Evaluación de Series Temporales: Precio en Chacra de Limón durante Eventos El Niño')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_pronostico_series_tiempo.png'), dpi=300)
plt.close()
print("[OK] Gráfico 3: 03_pronostico_series_tiempo.png")

# 4. Muestras de Hojas
sample_dir = 'data/sample_leaves'
sample_images = ['hoja_sana.jpg', 'tizon_bacteriano.jpg', 'mancha_marron.jpg']
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for idx, img_name in enumerate(sample_images):
    img_path = os.path.join(sample_dir, img_name)
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(f"Clase: {img_name.replace('.jpg', '').replace('_', ' ').title()}")
    axes[idx].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_muestras_patologias_hojas.png'), dpi=300)
plt.close()
print("[OK] Gráfico 4: 04_muestras_patologias_hojas.png")

# 5. Clustering de Provincias
df_prov = pd.read_csv('data/provincias_piura_riesgo.csv')
features_cluster = ['lluvia_max_historica_mm', 'pct_area_inundable', 'perdida_cosecha_nino_pct', 'deficit_drenaje_pct']
scaler = StandardScaler()
X_clus_scaled = scaler.fit_transform(df_prov[features_cluster])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_prov['cluster'] = kmeans.fit_predict(X_clus_scaled)
pca = PCA(n_components=2, random_state=42)
coords_pca = pca.fit_transform(X_clus_scaled)
df_prov['pca_1'] = coords_pca[:, 0]
df_prov['pca_2'] = coords_pca[:, 1]
colores_cluster = {0: '#d9534f', 1: '#f0ad4e', 2: '#5cb85c'}
etiquetas_cluster = {0: 'Vulnerabilidad Extrema', 1: 'Vulnerabilidad Alta', 2: 'Vulnerabilidad Moderada'}

fig, ax = plt.subplots(figsize=(9, 6))
for c_id in [0, 1, 2]:
    sub = df_prov[df_prov['cluster'] == c_id]
    ax.scatter(sub['pca_1'], sub['pca_2'], s=200, label=etiquetas_cluster[c_id], color=colores_cluster[c_id], edgecolors='k')
    for _, row in sub.iterrows():
        ax.annotate(row['provincia'], (row['pca_1'] + 0.08, row['pca_2'] + 0.08), fontsize=11, fontweight='bold')
ax.set_xlabel('Componente Principal 1 (Exposición y Lluvias Extremas)')
ax.set_ylabel('Componente Principal 2 (Déficit de Drenaje)')
ax.set_title(f'Zonificación K-Means de Provincias de Piura (Silhouette: {silhouette_score(X_clus_scaled, kmeans.labels_):.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_clustering_provincias_piura.png'), dpi=300)
plt.close()
print("[OK] Gráfico 5: 05_clustering_provincias_piura.png")
