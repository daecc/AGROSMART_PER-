# INFORME TÉCNICO DE DESARROLLO (AVANCE 50%)
## PROYECTO: AGROSMART PERÚ
### Sistema Inteligente para la Mitigación del Impacto del Fenómeno El Niño Costero en la Agricultura de la Costa Norte

---

**Curso:** Software Inteligente  
**Semana:** 14 (Avance 50%)  
**Ámbito Territorial:** Región Piura (Costa Norte del Perú)  
**Cultivos Foco:** Limón Sutil (*Citrus aurantifolia*) y Arroz Cáscara (*Oryza sativa*)  
**Cuadernos de Entrenamiento (Jupyter / Colab Modulares):**
1. [01_Prediccion_Supervisada_Rendimiento.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/01_Prediccion_Supervisada_Rendimiento.ipynb)
2. [02_Series_Temporales_Pronostico_Precios.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/02_Series_Temporales_Pronostico_Precios.ipynb)
3. [03_Clasificacion_Imagenes_Patologias_Hojas.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/03_Clasificacion_Imagenes_Patologias_Hojas.ipynb)
4. [04_Clustering_Vulnerabilidad_Provincias.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/04_Clustering_Vulnerabilidad_Provincias.ipynb)
5. [05_NLP_Resumen_Comunicados_ENFEN.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/05_NLP_Resumen_Comunicados_ENFEN.ipynb)  
**Fecha de Entrega:** Noviembre 2026  

---

## 1. INTRODUCCIÓN Y JUSTIFICACIÓN DEL PROBLEMA REAL

El Fenómeno de El Niño Costero (especialmente en sus eventos extraordinarios e históricos como 1983, 1998, 2017, 2023 y el actual evento 2026–2027 catalogado por el **ENFEN** con magnitud extraordinaria) constituye la principal amenaza hidroclimática para el Producto Bruto Interno (PBI) agropecuario en la costa norte del Perú.

En la región Piura (valles del Bajo Piura, Chira, San Lorenzo y Alto Piura), las anomalías térmicas en la región oceanográfica **Niño 1+2** superan los +2.6 °C, provocando lluvias torrenciales que saturan los suelos e inundan miles de hectáreas. Esto genera tres impactos críticos:
1. **Caída drástica en la producción y rendimiento:** En el cultivo de limón, el exceso hídrico asfixia las raíces y aborta la floración (con pérdidas registradas por el BCRP de hasta el 37.9% en 2017 y más del 40% en 2026), disparando los precios mayoristas en más de 300%. En el arroz, las inundaciones anegan campos listos para cosecha.
2. **Proliferación descontrolada de patógenos fúngicos y bacterianos:** La alta humedad relativa (>90%) y temperaturas nocturnas elevadas aceleran la propagación de enfermedades agresivas como el **Tizón Bacteriano** (*Xanthomonas oryzae*) y la **Mancha Marrón** (*Bipolaris oryzae*), devastando la rentabilidad de las familias agrícolas.
3. **Sobrecarga de información técnica no digerible:** Los agricultores y juntas de regantes reciben boletines meteorológicos y comunicados del ENFEN/SENAMHI de 15 a 30 páginas con densa jerga física y oceanográfica, lo que dificulta la toma de decisiones preventivas ágiles.

**AgroSmart Perú** nace como una solución de software inteligente integral con interfaz gráfica que articula cinco modelos avanzados de Inteligencia Artificial para predecir, proyectar, diagnosticar, zonificar y comunicar alertas agroclimáticas en tiempo real.

---

## 2. ORIGEN Y PREPARACIÓN DE DATOS (ARQUITECTURA DE DATOS HÍBRIDA)

Para garantizar rigor académico, transparencia y reproducibilidad ante el jurado, el dataset maestro se diseñó bajo una arquitectura de datos híbrida:

### 2.1. Series Históricas 100% Reales y Directas
Descargadas directamente desde los servidores de las entidades oficiales (almacenadas en `data/raw/` para auditoría):
1. **Banco Central de Reserva del Perú (BCRP):**
   * Serie mensual `PN01788AM`: Producción agrícola de **Limón** (miles de toneladas) de enero 2007 a septiembre 2026.
   * Serie mensual `PN01778AM`: Producción agrícola de **Arroz Cáscara** (miles de toneladas) de enero 2007 a septiembre 2026.
   * *Justificación de representatividad:* Piura concentra más del 55% de la producción nacional de limón; por ende, las variaciones de la serie nacional del BCRP reflejan directamente los colapsos de cosecha en los valles piuranos.
2. **National Oceanic and Atmospheric Administration (NOAA - EE.UU.):**
   * Registro satelital y de boyas de Temperatura Superficial del Mar (TSM) mensual en la región **Niño 1+2** ($0^\circ-10^\circ\text{S}, 90^\circ-80^\circ\text{W}$, masa oceánica frente a Piura y Tumbes) desde 1948 hasta 2026 (`data/raw/noaa_nino12.data`).
3. **Índice Costero El Niño (ICEN - IGP / ENFEN):**
   * Calculado formalmente siguiendo la metodología oficial del Instituto Geofísico del Perú (anomalía térmica de 3 meses móviles sobre la climatología oficial 1981–2010).
4. **Instituto Nacional de Estadística e Informática (INEI):**
   * Datos de las 8 provincias de Piura con sus códigos **UBIGEO oficiales (`2001` a `2008`)**, áreas agrícolas censales y valles principales.

### 2.2. Variables Agronómicas Calibradas con Reportes Oficiales
Dado que las entidades gubernamentales no proveen APIs REST públicas mensuales continuas para precipitación por caserío ni rendimiento parcelario:
* **Precipitación Mensual (`lluvia_mensual_mm`):** Modelada mediante correlación física con las anomalías de TSM y ajustada a los acumulados reales registrados por SENAMHI en las estaciones de Piura (Miraflores, Chulucanas, Partidor), oscilando entre 0–40 mm en épocas secas y **300–680 mm** en eventos Niño severos (2017, 2023, 2026).
* **Rendimientos y Precios en Chacra:** Calibrados a partir de los promedios de los anuarios "El Agro en Cifras" del MIDAGRI (12.5 $t/ha$ para limón y 8.8 $t/ha$ para arroz en Piura) y los boletines de precios mayoristas del SISAP.

El dataset consolidado se localiza en `data/tabla_maestra_agro_nino.csv` con **474 registros mensuales**.

---

## 3. PROCESO DE ENTRENAMIENTO Y RENDIMIENTO DE LOS MODELOS (JUPYTER / COLAB)

El ciclo de desarrollo de IA se encuentra completamente documentado y reproducible en el cuaderno interactivo **`notebooks/Entrenamiento_Modelos_AgroSmart.ipynb`**.

### 3.1. Módulo 1: Predicción Supervisada de Rendimiento Agrícola
* **Problema:** Regresión supervisada para estimar el rendimiento ($t/ha$).
* **Modelos Evaluados:** Regresión Lineal (Baseline) frente a `Random Forest Regressor` (100 árboles de decisión, profundidad máxima 6).
* **Partición de Datos:** 80% entrenamiento / 20% test con semilla fija.
* **Resultados y Métricas de Rendimiento:**

| Cultivo | Modelo | $R^2$ Score | MAE ($t/ha$) | RMSE ($t/ha$) |
| :--- | :--- | :---: | :---: | :---: |
| **Limón Sutil** | Regresión Lineal (Baseline) | 0.8841 | 0.421 | 0.512 |
| **Limón Sutil** | **Random Forest Regressor** | **0.9595** | **0.233** | **0.314** |
| **Arroz Cáscara**| Regresión Lineal (Baseline) | 0.8320 | 0.285 | 0.340 |
| **Arroz Cáscara**| **Random Forest Regressor** | **0.9385** | **0.159** | **0.202** |

*El análisis de importancia de características confirma que el índice **ICEN** y la **anomalía de TSM** explican más del 90% de la varianza en la merma del rendimiento.*

### 3.2. Módulo 2: Series Temporales de Precios en Chacra (Forecasting)
* **Problema:** Proyección temporal multivariada a 6–12 meses vista del precio al productor (S/ por kg).
* **Ingeniería de Características:** Retardos autorregresivos ($lag_1, lag_2, lag_3$) del precio, componentes estacionales ($\sin, \cos$ del mes) e **ICEN como variable exógena**.
* **Algoritmo:** `Gradient Boosting Regressor` (120 estimadores, tasa de aprendizaje 0.08).
* **Métricas en Conjunto de Prueba Temporal (últimos 36 meses):**
  * **Limón Sutil:** MAPE $= 8.11\%$, $\text{RMSE} = \text{S/ } 0.61\text{ / kg}$.
  * **Arroz Cáscara:** MAPE $= 5.00\%$, $\text{RMSE} = \text{S/ } 0.12\text{ / kg}$.

### 3.3. Módulo 3: Visión por Computadora (Diagnóstico Fitosanitario)
* **Problema:** Clasificación supervisada de fotografías de campo de arroz en 3 clases: *Hoja Sana*, *Tizón Bacteriano (Xanthomonas oryzae)* y *Mancha Marrón (Bipolaris oryzae)*.
* **Datos reales:** 1 424 fotografías de campo (300×300 px) del repositorio público de Hugging Face `girish787/riceLeafDataset` (recopilación de datasets de Kaggle tipo *Paddy Doctor*): 488 hojas sanas, 353 con tizón bacteriano y 583 con mancha marrón. Se eliminaron 127 duplicados exactos y 10 imágenes con etiqueta contradictoria (`src/vision_dataset.py`).
* **Partición sin fuga de información:** los casi-duplicados (hash perceptual dHash, distancia de Hamming ≤ 6) se agrupan y la división 80/20 se hace con `StratifiedGroupKFold`, de modo que dos fotos casi idénticas nunca quedan una en entrenamiento y otra en prueba.
* **Extracción de Características:** 123 descriptores por imagen: momentos de color RGB/HSV, histograma conjunto Tono×Saturación, histograma de brillo, fracción de píxeles verdes/amarillos/marrones/blanquecinos, textura LBP y orientación de gradientes.
* **Modelos comparados (validación cruzada por grupos de 5 particiones sobre el conjunto de entrenamiento):**

| Modelo | Accuracy CV | F1 macro CV |
| :--- | :---: | :---: |
| Regresión Logística (Baseline) | 0.790 ± 0.022 | 0.784 ± 0.023 |
| SVM RBF | 0.865 ± 0.022 | 0.861 ± 0.025 |
| Random Forest | 0.898 ± 0.029 | 0.897 ± 0.030 |
| **HistGradientBoosting** | **0.911 ± 0.031** | **0.909 ± 0.033** |

* **Desempeño del modelo final en el conjunto de prueba (285 fotos no vistas):** accuracy = **0.954**, F1 macro = **0.955** (Hoja Sana F1 0.944 · Tizón Bacteriano F1 0.964 · Mancha Marrón F1 0.957). La matriz de confusión se encuentra en `Documentos/graficos_evaluacion/06_matriz_confusion_vision.png`. El valor de validación cruzada (≈ 0.91) es la estimación más prudente del desempeño esperado.
* **Modelo exportado:** `models/vision_hojas_arroz.pkl`.
* **Limitaciones:** el modelo solo es válido para fotografías de hojas de arroz (no para limón). El 29% de las hojas sanas y el 15% de las de mancha marrón están fotografiadas sobre fondo blanco, pero ninguna de tizón bacteriano. En test, el modelo acierta 98.1% con fondo blanco y 94.8% en campo, por lo que no depende solo del fondo, pero una hoja con tizón sobre fondo blanco probablemente se clasificaría mal. Sus probabilidades tienden a ser muy altas y no deben leerse como certeza; las fotos no provienen de Piura, por lo que un despliegue real requeriría imágenes locales etiquetadas por SENASA.

### 3.4. Módulo 4: Clustering de Vulnerabilidad Provincial (No Supervisado)
* **Problema:** Segmentación territorial de las 8 provincias de Piura.
* **Justificación Matemática de $K=3$:**
  * Método del Codo (*Inertia*): Se observa la flexión de la curva en $k=3$.
  * Coeficiente de Silueta: **0.543** (máxima cohesión intra-cluster y separación inter-cluster).
* **Resultados del Clustering K-Means:**
  * **Cluster Rojo (Vulnerabilidad Extrema):** Morropón, Sechura, Piura y Sullana (pérdidas entre 39.5% y 51.2%, alto déficit de drenaje).
  * **Cluster Naranja (Vulnerabilidad Alta):** Ayabaca y Huancabamba (sierra: lluvias extremas, menor área inundable).
  * **Cluster Verde (Vulnerabilidad Moderada):** Paita y Talara.

### 3.5. Módulo 5: Resumen Inteligente de Comunicados Oficiales (NLP)
* **Problema:** Extracción no supervisada de conclusiones clave a partir de textos técnicos de 15–20 páginas del ENFEN y SENAMHI.
* **Algoritmo:** `TextRank` sobre matriz de similitud coseno de vectores `TF-IDF` con eliminación de stopwords en español.
* **Rendimiento:** Compresión textual del **76.4%**, aislando en segundos los párrafos de alerta agronómica y oceanográfica.

---

## 4. INTEGRACIÓN DE MODELOS EN LA APLICACIÓN WEB (`app.py`)

El flujo de integración sigue el estándar de la industria:
```
Cuaderno Jupyter / Colab
(Entrenamiento & Evaluación)
          │
          ▼  joblib.dump()
Carpeta models/
(rf_yield_*.pkl, ts_precio_*.pkl, vision_hojas_arroz.pkl, kmeans_provincias.pkl)
          │
          ▼  joblib.load()
Capa Backend (src/*.py)
(Inferencia en memoria RAM < 50ms)
          │
          ▼
Interfaz Web Streamlit (app.py)
(Dashboard interactivo para el usuario final)
```

**Una sola fuente de verdad:** los cuadernos exportan los modelos llamando a las mismas clases de `src/` que usa `app.py` (`AgroTimeSeriesForecaster.train_models()`, `PlantDiseaseClassifier.train_model()`, `VulnerabilityClusterer.train_clustering()`), de modo que cada `.pkl` tiene exactamente el formato y las variables que espera la aplicación. Al cargar, cada módulo valida el archivo (versión de características, variables o claves esperadas) y, si es incompatible o está corrupto, reentrena y lo vuelve a exportar en lugar de fallar. Los modelos se exportaron con `scikit-learn 1.9.1`, versión fijada en `requirements.txt`.

---

## 5. TIPO DE POBLACIÓN BENEFICIARIA

1. **Pequeños Agricultores y Juntas de Usuarios de Riego (Chira, San Lorenzo, Sechura):** Acceso a diagnósticos fotográficos directos y recomendaciones comprensibles.
2. **Gobierno Regional de Piura (GORE Piura) e INDECI:** Priorización geoespacial de motobombas y presupuesto de emergencia mediante el módulo de clustering.
3. **Ministerio de Desarrollo Agrario y Riego (MIDAGRI) y SENASA:** Distribución eficiente del FertiAbono y despliegue de brigadas fitosanitarias.
4. **Agrobanco y Aseguradoras Agrícolas:** Evaluación técnica del riesgo para el Seguro Agrario Catastrófico (SAC).

---

## 6. VENTAJAS DE AGROSMART PERÚ FRENTE A OTROS SOFTWARES

| Criterio | Software Agrícola Convencional | AgroSmart Perú |
| :--- | :--- | :--- |
| **Integración Climática** | Solo registra datos pasados. | **Predictivo:** Conecta activamente las anomalías del mar (ICEN/NOAA) con el rendimiento futuro. |
| **Enfoque Territorial** | Software genérico sin calibración local. | **100% Calibrado para el Perú:** Específico para las 8 provincias de Piura y cultivos vulnerables. |
| **Autonomía Operativa** | Requiere internet y APIs de pago. | **Operación Local/Offline:** Modelos preentrenados serializados listos para trabajo en campo. |
| **Procesamiento de Alertas** | Lectura manual de PDFs de 30 páginas. | **NLP Automatizado:** Ficha ejecutiva en 5 segundos con directrices de acción directa. |
| **Costo y Despliegue** | Licencias privadas costosas en dólares. | **Código Abierto y Python Nativo:** Desplegable sin costo en cualquier navegador web. |

---

## 7. CRONOGRAMA Y ESTADO DE AVANCE (50%)

- [x] **Semana 14 (Avance 50% - CUMPLIDO):**
  - Dataset maestro real con BCRP y NOAA integrado.
  - Cuaderno Jupyter formal de entrenamiento paso a paso con métricas ($R^2 > 0.93$, $\text{MAPE} < 9\%$).
  - 5 modelos entrenados y exportados a `models/`.
  - Aplicación Streamlit (`app.py`) 100% funcional y navegable.
  - Manual de instalación y de usuario redactado.
- [ ] **Semana 15 (Presentación Final 100%):**
  - Incorporación de mapas geográficos interactivos con `Folium`.
  - Diapositivas de exposición final del proyecto.
